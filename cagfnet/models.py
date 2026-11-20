"""CAGF-Net and the comparison models.

Blocks (Fig. 1 of the paper):
  A  CT encoder       - 4 x [Conv3x3, BN, ReLU] (2x2 max-pool after the first three),
                        10x10x64 map -> 100 tokens + learned positional code
  B  audio encoder    - 2-layer 1D CNN over (AC, BC, mask) curves (dilation 2 in layer 2),
                        concatenated with 11 clinical features, MLP -> 4 tokens
  C  cross-attention  - bidirectional MHA (H=4, d_k=16) with Add&LN and FFN(128)
  D  gated fusion     - g = sigma(W_g[h_t;h_a]+b_g), h = g*tanh(W_t h_t) + (1-g)*tanh(W_a h_a)
  E  head             - Dropout -> FC(64) -> GELU -> Dropout -> FC(1) -> logit z
Modality-dropout replaces one modality by learned surrogate tokens with probability p_md.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CTEncoder(nn.Module):
    def __init__(self, channels=(16, 32, 64, 64), d_model=64, roi=80):
        super().__init__()
        layers, c_in = [], 2
        for i, c in enumerate(channels):
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            c_in = c
        self.cnn = nn.Sequential(*layers)
        side = roi // 2 ** (len(channels) - 1)
        self.n_tokens = side * side
        self.proj = nn.Linear(c_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):                                   # x: (B, 2, 80, 80)
        f = self.cnn(x)                                     # (B, 64, 10, 10)
        tokens = f.flatten(2).transpose(1, 2)               # (B, 100, 64)
        return self.proj(tokens) + self.pos


class AudioEncoder(nn.Module):
    def __init__(self, channels=32, n_freq=7, n_clinical=11, d_model=64, n_tokens=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2), nn.BatchNorm1d(channels), nn.GELU(),
        )
        self.n_tokens, self.d_model = n_tokens, d_model
        self.mlp = nn.Sequential(
            nn.Linear(channels * n_freq + n_clinical, 128), nn.GELU(),
            nn.Linear(128, n_tokens * d_model),
        )

    def forward(self, curves, clinical):                    # (B,3,7), (B,11)
        h = self.conv(curves).flatten(1)                    # (B, 224)
        t = self.mlp(torch.cat([h, clinical], dim=1))
        return t.view(-1, self.n_tokens, self.d_model)      # (B, 4, 64)


class CrossAttentionBlock(nn.Module):
    """T_t' = LN(T_t + MHA(T_t, T_a, T_a)),  T_a' = LN(T_a + MHA(T_a, T_t, T_t)),
    T_m'' = LN(T_m' + W_4 GELU(W_3 T_m'))."""

    def __init__(self, d_model=64, n_heads=4, ffn_hidden=128, dropout=0.1):
        super().__init__()
        self.att_t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.att_a = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_t1, self.ln_a1 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ln_t2, self.ln_a2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn_t = nn.Sequential(nn.Linear(d_model, ffn_hidden), nn.GELU(), nn.Linear(ffn_hidden, d_model))
        self.ffn_a = nn.Sequential(nn.Linear(d_model, ffn_hidden), nn.GELU(), nn.Linear(ffn_hidden, d_model))
        self.last_attn = None

    def forward(self, tt, ta):
        at, w_t = self.att_t(tt, ta, ta, need_weights=True, average_attn_weights=True)
        aa, _ = self.att_a(ta, tt, tt, need_weights=False)
        self.last_attn = w_t.detach()                       # (B, 100, 4): CT region -> audio token
        tt = self.ln_t1(tt + at)
        ta = self.ln_a1(ta + aa)
        tt = self.ln_t2(tt + self.ffn_t(tt))
        ta = self.ln_a2(ta + self.ffn_a(ta))
        return tt, ta


class GatedFusion(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)
        self.wt = nn.Linear(d_model, d_model)
        self.wa = nn.Linear(d_model, d_model)
        self.last_gate = None

    def forward(self, ht, ha):
        g = torch.sigmoid(self.gate(torch.cat([ht, ha], dim=1)))
        self.last_gate = g.detach()
        return g * torch.tanh(self.wt(ht)) + (1 - g) * torch.tanh(self.wa(ha))


def _head(d_in, d_hidden, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(d_hidden, 1))


class CAGFNet(nn.Module):
    """Configurable fusion network; flags give CAGF-Net, its ablations and early fusion."""

    def __init__(self, cfg: dict, use_cross_attn=True, use_gate=True, modality_dropout=None):
        super().__init__()
        m = cfg["model"]
        d = m["d_model"]
        self.ct = CTEncoder(m["ct_channels"], d, cfg["data"]["roi_size"])
        self.audio = AudioEncoder(m["audio_channels"], 7, 11, d, m["audio_tokens"])
        self.use_cross_attn, self.use_gate = use_cross_attn, use_gate
        self.p_md = m["modality_dropout"] if modality_dropout is None else modality_dropout
        self.xattn = CrossAttentionBlock(d, m["n_heads"], m["ffn_hidden"]) if use_cross_attn else None
        self.fusion = GatedFusion(d) if use_gate else nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.head = _head(d, d, m["dropout"])
        self.m_t = nn.Parameter(torch.zeros(1, self.ct.n_tokens, d))   # learned surrogate tokens
        self.m_a = nn.Parameter(torch.zeros(1, m["audio_tokens"], d))
        nn.init.trunc_normal_(self.m_t, std=0.02)
        nn.init.trunc_normal_(self.m_a, std=0.02)

    def forward(self, ct, curves, clinical, missing: str | None = None):
        tt = self.ct(ct)
        ta = self.audio(curves, clinical)
        b = tt.shape[0]
        if missing == "ct":
            tt = self.m_t.expand(b, -1, -1)
        elif missing == "audio":
            ta = self.m_a.expand(b, -1, -1)
        elif self.training and self.p_md > 0:
            drop = torch.rand(b, device=tt.device) < self.p_md
            which_ct = torch.rand(b, device=tt.device) < 0.5
            dt = (drop & which_ct).view(b, 1, 1)
            da = (drop & ~which_ct).view(b, 1, 1)
            tt = torch.where(dt, self.m_t.expand(b, -1, -1), tt)
            ta = torch.where(da, self.m_a.expand(b, -1, -1), ta)
        if self.xattn is not None:
            tt, ta = self.xattn(tt, ta)
        ht, ha = tt.mean(1), ta.mean(1)
        h = self.fusion(ht, ha) if self.use_gate else self.fusion(torch.cat([ht, ha], 1))
        return self.head(h).squeeze(1)


class SingleModality(nn.Module):
    def __init__(self, cfg: dict, modality: str):
        super().__init__()
        m = cfg["model"]
        d = m["d_model"]
        self.modality = modality
        if modality == "ct":
            self.enc = CTEncoder(m["ct_channels"], d, cfg["data"]["roi_size"])
        else:
            self.enc = AudioEncoder(m["audio_channels"], 7, 11, d, m["audio_tokens"])
        self.head = _head(d, d, m["dropout"])

    def forward(self, ct, curves, clinical, missing=None):
        t = self.enc(ct) if self.modality == "ct" else self.enc(curves, clinical)
        return self.head(t.mean(1)).squeeze(1)


MODEL_NAMES = {
    "ct_only": "CT only",
    "audio_only": "Audio only",
    "late_fusion": "Late fusion (Bayesian)",
    "early_fusion": "Early fusion (concatenation)",
    "cagf_net": "CAGF-Net (proposed)",
    "cagf_no_xattn": "CAGF-Net w/o cross-attention",
    "cagf_no_gate": "CAGF-Net w/o gate (concat)",
    "cagf_no_moddrop": "CAGF-Net w/o modality-dropout",
}


def build_model(name: str, cfg: dict) -> nn.Module:
    if name == "ct_only":
        return SingleModality(cfg, "ct")
    if name == "audio_only":
        return SingleModality(cfg, "audio")
    if name == "early_fusion":
        return CAGFNet(cfg, use_cross_attn=False, use_gate=False, modality_dropout=0.0)
    if name == "cagf_net":
        return CAGFNet(cfg)
    if name == "cagf_no_xattn":
        return CAGFNet(cfg, use_cross_attn=False)
    if name == "cagf_no_gate":
        return CAGFNet(cfg, use_gate=False)
    if name == "cagf_no_moddrop":
        return CAGFNet(cfg, modality_dropout=0.0)
    raise ValueError(f"unknown model {name}")
