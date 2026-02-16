import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2, act=nn.SiLU) -> nn.Sequential:
    """
    Tiny MLP helper.
    """
    assert n_layers >= 2
    layers = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(act())
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def safe_cholesky_solve(K: torch.Tensor, y: torch.Tensor, base_jitter: float = 1e-4, max_tries: int = 8):
    """
    Robust SPD solve for (K) x = y.
    Returns: x, jitter_used
    """
    # Symmetrization
    K = 0.5 * (K + K.transpose(-1, -2))

    if not torch.isfinite(K).all():
        raise FloatingPointError("K contains NaN/Inf before Cholesky.")

    B, N, _ = K.shape
    # y shape: (B, N, D) or (B, N)
    if y.dim() == 2:
        y = y.unsqueeze(-1)  # (B, N, 1)

    eye = torch.eye(N, device=K.device, dtype=K.dtype).unsqueeze(0)
    jitter = float(base_jitter)
    last_err = None

    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(K + jitter * eye)
            # cholesky_solve computes K^{-1} y
            x = torch.cholesky_solve(y, L)
            return x.squeeze(-1), jitter, L  # Return L for potential reuse if needed
        except RuntimeError as e:
            last_err = e
            jitter *= 10.0

    raise RuntimeError(f"Cholesky failed after {max_tries} tries. Last error: {last_err}")


class CD2_RBFKI(nn.Module):
    """
    Conditional Deep Ordinary Kriging (CDK-OK).

    Theory:
    1. Z-score normalization (handled by dataset).
    2. Variance Matching:
       K_total = eta * K_param + (1 - eta) * K_res_norm
       This ensures diag(K_total) == 1.
    3. Ordinary Kriging Solver:
       Estimates global mean 'mu' and weights 'alpha' to satisfy sum(w)=1.
       Prediction = Kernel^T * alpha + mu
    """

    def __init__(
            self,
            d_e: int = 32,
            d_c: int = 32,
            hidden: int = 64,
            rank_r: int = 8,
            nugget: float = 1e-4,
            chunk_q: int = 1024,
            eps_pd: float = 1e-6,
            rho_min: float = 0.01,
            rho_max: float = 5.0,
            L_diag_min: float = 0.02,
            L_diag_max: float = 8.0,
            L_offdiag_max: float = 3.0,

            # ---------------- Ablations ----------------
            ablate_no_L: bool = False,
            ablate_fixed_rhop: bool = False,
            fixed_rho: float = 1.0,
            fixed_p: float = 2.0,
            ablate_no_residual: bool = False,
            ablate_fixed_beta: bool = False,
            fixed_beta: float = 0.5,  # In OK version, this acts as the mixing eta
    ):
        super().__init__()
        self.rank_r = int(rank_r)
        self.nugget = float(nugget)
        self.chunk_q = int(chunk_q)
        self.eps_pd = float(eps_pd)

        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.L_diag_min = float(L_diag_min)
        self.L_diag_max = float(L_diag_max)
        self.L_offdiag_max = float(L_offdiag_max)

        self.ablate_no_L = bool(ablate_no_L)
        self.ablate_fixed_rhop = bool(ablate_fixed_rhop)
        self.fixed_rho = float(fixed_rho)
        self.fixed_p = float(fixed_p)
        self.ablate_no_residual = bool(ablate_no_residual)
        self.ablate_fixed_beta = bool(ablate_fixed_beta)
        self.fixed_beta = float(fixed_beta)

        # DeepSets encoder
        self.phi = make_mlp(in_dim=3, hidden_dim=hidden, out_dim=d_e, n_layers=2)
        self.rho = make_mlp(in_dim=d_e, hidden_dim=hidden, out_dim=d_c, n_layers=2)

        # Heads from c
        self.head_L = make_mlp(in_dim=d_c, hidden_dim=hidden, out_dim=3, n_layers=2)
        self.head_rho = make_mlp(in_dim=d_c, hidden_dim=hidden, out_dim=1, n_layers=2)
        self.head_p = make_mlp(in_dim=d_c, hidden_dim=hidden, out_dim=1, n_layers=2)

        # Convex combination weight 'eta' (replaces simple beta)
        # eta * K_param + (1-eta) * K_res
        self.head_eta = make_mlp(in_dim=d_c, hidden_dim=hidden, out_dim=1, n_layers=2)

        # g(s,c) low-rank feature map
        self.g_mlp = make_mlp(in_dim=2 + d_c, hidden_dim=hidden, out_dim=self.rank_r, n_layers=2)

    def _build_L(self, a1, a2, a3):
        B = a1.shape[0]
        L11 = (F.softplus(a1) + self.eps_pd).clamp(self.L_diag_min, self.L_diag_max)
        L21 = a2.clamp(-self.L_offdiag_max, self.L_offdiag_max)
        L22 = (F.softplus(a3) + self.eps_pd).clamp(self.L_diag_min, self.L_diag_max)
        L = torch.zeros(B, 2, 2, device=a1.device, dtype=a1.dtype)
        L[:, 0, 0] = L11;
        L[:, 1, 0] = L21;
        L[:, 1, 1] = L22
        return L

    def _encode_context(self, coords, values):
        inp = torch.cat([coords, values.unsqueeze(-1)], dim=-1)
        e = self.phi(inp)
        c = self.rho(e.mean(dim=1))
        return c

    def _kernel_rbf(self, X, Y, L, rho, p):
        Xt = torch.bmm(X, L.transpose(1, 2))
        Yt = torch.bmm(Y, L.transpose(1, 2))
        dist = torch.cdist(Xt, Yt)
        z = dist / (rho + 1e-12)
        return torch.exp(-(z ** p))

    def forward(self, coords, values, query_coords, return_aux=False):
        B, N, _ = coords.shape
        _, M, _ = query_coords.shape
        device = coords.device

        # --- 1) Context ---
        c = self._encode_context(coords, values)

        # --- 2) Kernel Parameters ---
        a = self.head_L(c)
        L = self._build_L(a[:, 0], a[:, 1], a[:, 2])
        if self.ablate_no_L:
            L = torch.eye(2, device=device).expand(B, -1, -1)

        rho = self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(self.head_rho(c))
        p = 0.5 + 1.5 * torch.sigmoid(self.head_p(c))

        if self.ablate_fixed_rhop:
            rho = torch.full_like(rho, self.fixed_rho)
            p = torch.full_like(p, self.fixed_p)

        # eta: convex combination weight
        eta = torch.sigmoid(self.head_eta(c))  # (B, 1)
        if self.ablate_fixed_beta:  # Use this flag to fix eta
            eta = torch.full_like(eta, self.fixed_beta)
        if self.ablate_no_residual:  # If no residual, eta must be 1.0 (pure param)
            eta = torch.ones_like(eta)

        rho_, p_, eta_ = rho.view(B, 1, 1), p.view(B, 1, 1), eta.view(B, 1, 1)

        # --- 3) Parametric Kernel (RBF) ---
        # Diag(K_rbf) is always 1.0
        coords64 = coords.double()
        L64 = L.double()
        rho64, p64 = rho_.double(), p_.double()

        K_rbf_SS = self._kernel_rbf(coords64, coords64, L=L64, rho=rho64, p=p64)

        # --- 4) Normalized Residual Kernel ---
        # K_res = <g, g>. To make Diag(K_res) == 1, we normalize features g.
        use_residual = (eta.min().item() < 1.0)
        gS_norm = None

        if use_residual:
            cS = c[:, None, :].expand(B, N, c.shape[-1])
            gS = torch.tanh(self.g_mlp(torch.cat([coords, cS], dim=-1)))  # (B,N,r)
            # L2 Normalize feature dimension to ensure dot product self-similarity is 1
            # Note: tanh limits value, but we strictly want <g,g>=1 on diag.
            # Normalizing g is the cleanest way.
            gS_norm = F.normalize(gS, p=2, dim=-1)  # (B, N, r)

            # K_res = gS @ gS.T -> (B, N, N)
            K_res_SS = torch.bmm(gS_norm.double(), gS_norm.double().transpose(1, 2))
        else:
            K_res_SS = 0.0

        # --- 5) Total Kernel (Convex Combination) ---
        # K_total = eta * K_rbf + (1-eta) * K_res
        # Diag(K_total) = eta * 1 + (1-eta) * 1 = 1. Matches Z-score variance.
        eta64 = eta_.double()
        K_SS = eta64 * K_rbf_SS + (1.0 - eta64) * K_res_SS

        # --- 6) Ordinary Kriging Solver (Differentiable) ---
        # Solve: (K + lambda I) * v_y = y
        # Solve: (K + lambda I) * v_1 = 1
        y64 = values.double()
        ones64 = torch.ones((B, N), dtype=torch.float64, device=device)

        # We reuse the Cholesky factor inside safe_cholesky_solve if possible,
        # but for simplicity/safety we call it twice or modify the helper.
        # Let's stack y and 1 to solve in one go for efficiency: (B, N, 2)
        rhs = torch.stack([y64, ones64], dim=-1)  # (B, N, 2)

        sol, used_jitter, _ = safe_cholesky_solve(K_SS, rhs, base_jitter=self.nugget)
        v_y = sol[..., 0]  # (B, N)
        v_1 = sol[..., 1]  # (B, N)

        # Compute GLS Mean: mu = (1^T v_y) / (1^T v_1)
        top = v_y.sum(dim=1, keepdim=True)  # (B, 1)
        bot = v_1.sum(dim=1, keepdim=True)  # (B, 1)
        mu_gls = top / (bot + 1e-9)  # (B, 1)

        # Compute OK Weights: alpha = v_y - mu * v_1
        alpha = v_y - mu_gls * v_1  # (B, N)

        # --- 7) Prediction ---
        # y_hat(q) = K(q, X)^T * alpha + mu

        # A. Residual Part (accelerated)
        pred_res = 0.0
        if use_residual:
            # We need to project alpha onto gS_norm
            # w_proj = sum(alpha_i * g_i)
            w_proj = (alpha.float().unsqueeze(-1) * gS_norm).sum(dim=1)  # (B, r)

            cQ = c[:, None, :].expand(B, M, c.shape[-1])
            gQ = torch.tanh(self.g_mlp(torch.cat([query_coords, cQ], dim=-1)))
            gQ_norm = F.normalize(gQ, p=2, dim=-1)

            # K_res(q, X) * alpha = (1-eta) * <gQ, w_proj>
            # Don't forget the (1-eta) weight!
            pred_res = (1.0 - eta) * (gQ_norm * w_proj[:, None, :]).sum(dim=-1)

        # B. Parametric Part
        pred_rbf = torch.zeros(B, M, device=device, dtype=torch.float32)
        chunk = max(1, self.chunk_q)
        q64_all = query_coords.double()

        for start in range(0, M, chunk):
            end = min(M, start + chunk)
            q_sub = q64_all[:, start:end, :]
            # K_rbf(q, X)
            k_rbf_sub = self._kernel_rbf(q_sub, coords64, L=L64, rho=rho64, p=p64).float()
            # Weight by alpha and eta
            # Term: eta * K_rbf * alpha
            pred_rbf[:, start:end] = (eta * k_rbf_sub * alpha.float()[:, None, :]).sum(dim=-1)

        # C. Combine + Mean Shift
        # y_hat = (eta*K_rbf + (1-eta)K_res) * alpha + mu
        pred = pred_rbf + pred_res + mu_gls.float()

        if return_aux:
            aux = {
                "c": c, "eta": eta, "mu": mu_gls,
                "L": L, "rho": rho, "p": p,
                "used_jitter": used_jitter,
                "alpha": alpha.float()
            }
            return pred, aux
        return pred