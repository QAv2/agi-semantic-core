"""W3 network (docs/W3_LOOP_PROTOCOL.md §1.2). The classes sinusoid, Res, Attn and UNet are lifted
VERBATIM from windows/power/w3_budget_check.py (the code timed on the T4; a test asserts the lift).
W3UNet adds two things that change no computation (a test asserts forward equality bit for bit):
  (a) the activations at the seven sites of §3.1, each mean-pooled over space;
  (b) a forward pass that takes the label embedding e directly instead of the step t, used by
      M-drop's learned no-label vector e-null and by the label sham.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- verbatim from windows/power/w3_budget_check.py ----------------------------------------
def sinusoid(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Res(nn.Module):
    def __init__(self, a, b, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, a), nn.Conv2d(a, b, 3, 1, 1)
        self.t = nn.Linear(tdim, b) if tdim else None
        self.n2, self.c2 = nn.GroupNorm(8, b), nn.Conv2d(b, b, 3, 1, 1)
        self.skip = nn.Conv2d(a, b, 1) if a != b else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        if self.t is not None:
            h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n, self.qkv, self.o = nn.GroupNorm(8, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x, _=None):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, c, h * w).unbind(1)
        a = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        return x + self.o((v @ a.transpose(1, 2)).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, cin=1, C=64, mult=(1, 2, 2), clock=True):
        super().__init__()
        tdim = 4 * C if clock else 0
        self.C, self.clock = C, clock
        self.temb = nn.Sequential(nn.Linear(C, tdim), nn.SiLU(), nn.Linear(tdim, tdim)) if clock else None
        self.inp = nn.Conv2d(cin, C, 3, 1, 1)
        chs, self.down, ch = [C], nn.ModuleList(), C
        for i, m in enumerate(mult):
            for _ in range(2):
                blk = nn.ModuleList([Res(ch, C * m, tdim)])
                ch = C * m
                if i == len(mult) - 1:
                    blk.append(Attn(ch))
                self.down.append(blk)
                chs.append(ch)
            if i < len(mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, 2, 1)]))
                chs.append(ch)
        self.mid = nn.ModuleList([Res(ch, ch, tdim), Attn(ch), Res(ch, ch, tdim)])
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(3):
                blk = nn.ModuleList([Res(ch + chs.pop(), C * m, tdim)])
                ch = C * m
                if i == len(mult) - 1:
                    blk.append(Attn(ch))
                self.up.append(blk)
            if i > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2), nn.Conv2d(ch, ch, 3, 1, 1)]))
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, cin, 3, 1, 1))

    def forward(self, x, t):
        temb = self.temb(sinusoid(t, self.C)) if self.clock else None
        h, hs = self.inp(x), []
        hs.append(h)
        for blk in self.down:
            if isinstance(blk[0], nn.Conv2d):
                h = blk[0](h)
            else:
                for layer in blk:
                    h = layer(h, temb)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, temb)
        for blk in self.up:
            if isinstance(blk[0], nn.Upsample):
                h = blk[1](blk[0](h))
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                for layer in blk:
                    h = layer(h, temb)
        return self.out(h)


# ---- end of the verbatim lift ---------------------------------------------------------------


class W3UNet(UNet):
    """UNet + site features + the embedding path. drop=True adds M-drop's learned e-null."""

    def __init__(self, cin=3, C=64, mult=(1, 2, 2), clock=True, drop=False):
        super().__init__(cin=cin, C=C, mult=mult, clock=clock)
        assert clock or not drop
        self.null_emb = nn.Parameter(torch.zeros(4 * C)) if drop else None
        self.n_levels = len(mult)
        # site indices: the last block before each downsample / upsample, and the last block
        self._enc_sites, self._dec_sites = [], []
        for i, blk in enumerate(self.down):
            if isinstance(blk[0], nn.Conv2d):
                self._enc_sites.append(i - 1)
        self._enc_sites.append(len(self.down) - 1)
        for i, blk in enumerate(self.up):
            if isinstance(blk[0], nn.Upsample):
                self._dec_sites.append(i - 1)
        self._dec_sites.append(len(self.up) - 1)
        self.site_names = tuple(f"enc{32 >> k}" for k in range(len(mult))) + ("mid",) + \
            tuple(f"dec{32 >> (len(mult) - 1 - k)}" for k in range(len(mult)))

    def embed(self, t):
        """The label embedding e(t) that every residual block receives."""
        return self.temb(sinusoid(t, self.C))

    def label(self, t=None, emb=None, null_mask=None):
        if not self.clock:
            return None
        temb = emb if emb is not None else self.embed(t)
        if null_mask is not None:
            temb = torch.where(null_mask[:, None], self.null_emb.to(temb.dtype).expand_as(temb), temb)
        return temb

    def forward(self, x, t=None, emb=None, null_mask=None, feats=False):
        """The original UNet.forward, step for step, with the label from label() and the
        site activations recorded when feats=True (returns (out, {site: (B, ch)}))."""
        temb = self.label(t, emb, null_mask)
        rec = {}
        h, hs = self.inp(x), []
        hs.append(h)
        for i, blk in enumerate(self.down):
            if isinstance(blk[0], nn.Conv2d):
                h = blk[0](h)
            else:
                for layer in blk:
                    h = layer(h, temb)
            hs.append(h)
            if feats and i in self._enc_sites:
                rec[self.site_names[self._enc_sites.index(i)]] = h.mean(dim=(2, 3))
        for layer in self.mid:
            h = layer(h, temb)
        if feats:
            rec["mid"] = h.mean(dim=(2, 3))
        for i, blk in enumerate(self.up):
            if isinstance(blk[0], nn.Upsample):
                h = blk[1](blk[0](h))
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                for layer in blk:
                    h = layer(h, temb)
            if feats and i in self._dec_sites:
                rec[self.site_names[self.n_levels + 1 + self._dec_sites.index(i)]] = h.mean(dim=(2, 3))
        out = self.out(h)
        return (out, rec) if feats else out


def make_model(tag, C=64, mult=(1, 2, 2), seed=None):
    """tag in {on, drop, off} (an untrained twin is make_model(...) never trained). Weights are
    initialised under `seed` (the TRAIN seed)."""
    if seed is not None:
        torch.manual_seed(seed)
    return W3UNet(cin=3, C=C, mult=mult, clock=(tag != "off"), drop=(tag == "drop"))
