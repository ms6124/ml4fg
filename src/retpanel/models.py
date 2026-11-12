"""Classifiers and the concrete (Gumbel-softmax) gene selector with an optional
donor-adversarial head trained through a gradient-reversal layer."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


def device() -> torch.device:
    # MPS is opt-in: torch 2.15's MPS linear/matmul kernels crash on macOS 14 (NSInvalidArgumentException).
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("RETPANEL_MPS") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Linear baseline
# ---------------------------------------------------------------------------
def fit_logreg(Z: np.ndarray, y: np.ndarray, C: float = 1.0, max_iter: int = 300, seed: int = 0):
    """L2 multinomial logistic regression (lbfgs). Inputs are z-scored features."""
    clf = LogisticRegression(C=C, max_iter=max_iter, tol=1e-3, random_state=seed)
    clf.fit(Z, y)
    return clf


def predict_proba_full(clf, Z: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba padded to all classes (some may be missing from a training fold)."""
    p = clf.predict_proba(Z)
    out = np.zeros((len(Z), n_classes), dtype=np.float32)
    out[:, clf.classes_] = p
    return out


# ---------------------------------------------------------------------------
# Concrete selector
# ---------------------------------------------------------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class ConcreteSelector(nn.Module):
    """k selector rows, each a categorical distribution over G genes (Abid et al. 2019)."""

    def __init__(self, n_genes: int, k: int):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(k, n_genes) * 0.01)
        self.temperature = 10.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.logits).clamp_(1e-10, 1 - 1e-10)
            g = -torch.log(-torch.log(u))
            w = F.softmax((self.logits + g) / self.temperature, dim=1)
            # .contiguous(): strided matmul operands crash the MPS backend on macOS 14
            return x @ w.t().contiguous()
        return x[:, self.logits.argmax(1)]

    def duplicate_penalty(self) -> torch.Tensor:
        """Penalise total selection mass > 1 on any gene, discouraging repeated picks."""
        p = F.softmax(self.logits / self.temperature, dim=1)
        return F.relu(p.sum(0) - 1).sum()

    def panel(self) -> np.ndarray:
        """Distinct gene indices: argmax per row; duplicates replaced greedily by each
        row's next-best unused gene."""
        L = self.logits.detach().cpu().numpy()
        order = np.argsort(-L.max(1))  # most confident rows choose first
        chosen, used = [], set()
        for r in order:
            for g in np.argsort(-L[r]):
                if g not in used:
                    used.add(int(g))
                    chosen.append(int(g))
                    break
        return np.array(chosen)

    def n_unique_argmax(self) -> int:
        return int(torch.unique(self.logits.argmax(1)).numel())


def mlp(d_in: int, d_out: int, hidden: int = 256, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, d_out),
    )


class PanelNet(nn.Module):
    def __init__(self, n_genes, k, n_classes, n_donors=0, hidden=256):
        super().__init__()
        self.selector = ConcreteSelector(n_genes, k)
        self.clf = mlp(k, n_classes, hidden)
        # Donor head sees selected genes *and* the true class, so it is penalised only for
        # donor signal beyond what is explained by class composition.
        self.adv = mlp(k + n_classes, n_donors, 128) if n_donors > 0 else None
        self.n_classes = n_classes


@dataclass
class ConcreteConfig:
    k: int = 50
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 1e-3
    selector_lr: float = 5e-2   # selector logits need to move far: own, larger step size
    t_start: float = 2.0
    anneal_frac: float = 0.75   # anneal over this fraction of steps, then train head on ~hard selection
    t_end: float = 0.01
    adv_lambda: float = 0.0     # 0 disables the donor-adversarial term
    dup_weight: float = 1.0
    hidden: int = 256
    balanced: bool = True       # inverse-frequency class weights (macro metrics reward rare classes)
    seed: int = 0


def train_concrete(Z: np.ndarray, y: np.ndarray, donors: np.ndarray | None, n_classes: int,
                   cfg: ConcreteConfig, log_every: int = 5) -> tuple[PanelNet, list[dict]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = device()
    use_adv = cfg.adv_lambda > 0 and donors is not None
    d_codes = None
    n_donors = 0
    if use_adv:
        uniq, d_codes = np.unique(donors, return_inverse=True)
        n_donors = len(uniq)
    net = PanelNet(Z.shape[1], cfg.k, n_classes, n_donors, cfg.hidden).to(dev)
    head_params = [p for n, p in net.named_parameters() if not n.startswith("selector.")]
    opt = torch.optim.Adam([{"params": head_params, "lr": cfg.lr},
                            {"params": [net.selector.logits], "lr": cfg.selector_lr}])

    Xt = torch.from_numpy(Z).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    cw = None
    if cfg.balanced:
        cnt = np.bincount(y, minlength=n_classes).astype(np.float32)
        cw = np.where(cnt > 0, cnt.sum() / (n_classes * np.maximum(cnt, 1)), 0.0)
        cw = torch.from_numpy(cw.astype(np.float32)).to(dev)
    dt = torch.from_numpy(d_codes.astype(np.int64)).to(dev) if use_adv else None
    n = len(Z)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    step = 0
    history = []
    for ep in range(cfg.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        agg = {"cls": 0.0, "adv": 0.0, "acc": 0.0, "adv_acc": 0.0}
        for b in range(steps_per_epoch):
            # exponential temperature annealing per step
            frac = min(step / max(cfg.anneal_frac * total, 1), 1.0)
            net.selector.temperature = cfg.t_start * (cfg.t_end / cfg.t_start) ** frac
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            s = net.selector(Xt[idx])
            logits = net.clf(s)
            loss_cls = F.cross_entropy(logits, yt[idx], weight=cw)
            loss = loss_cls + cfg.dup_weight * net.selector.duplicate_penalty() / cfg.k
            if use_adv:
                # lambda ramp-up (Ganin et al. 2016) to keep early training stable
                p = step / total
                lam = cfg.adv_lambda * (2 / (1 + math.exp(-10 * p)) - 1)
                inp = torch.cat([GradReverse.apply(s, lam), F.one_hot(yt[idx], n_classes).float()], 1)
                dl = net.adv(inp)
                loss_adv = F.cross_entropy(dl, dt[idx])
                loss = loss + loss_adv
                agg["adv"] += loss_adv.item()
                agg["adv_acc"] += (dl.argmax(1) == dt[idx]).float().mean().item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            agg["cls"] += loss_cls.item()
            agg["acc"] += (logits.argmax(1) == yt[idx]).float().mean().item()
            step += 1
        rec = {k: v / steps_per_epoch for k, v in agg.items()}
        rec.update(epoch=ep, temperature=net.selector.temperature, n_unique=net.selector.n_unique_argmax())
        history.append(rec)
        if ep % log_every == 0 or ep == cfg.epochs - 1:
            print(f"  ep {ep:3d} T={rec['temperature']:.3f} cls={rec['cls']:.3f} acc={rec['acc']:.3f} "
                  f"adv={rec['adv']:.3f} adv_acc={rec['adv_acc']:.3f} unique={rec['n_unique']}", flush=True)
    return net, history


@torch.no_grad()
def predict_concrete(net: PanelNet, Z: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    net.eval()
    dev = next(net.parameters()).device
    out = []
    for i in range(0, len(Z), batch_size):
        x = torch.from_numpy(Z[i:i + batch_size]).to(dev)
        out.append(F.softmax(net.clf(net.selector(x)), 1).cpu().numpy())
    return np.concatenate(out)
