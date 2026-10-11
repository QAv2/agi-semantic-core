#!/usr/bin/env python3
"""W3 budget check (§4 of the design check): size the candidate denoisers and estimate T4 time.
Lane law: BEFORE registration, BEFORE build. Counts FLOPs with torch's FlopCounterMode on one
training step; times are an extrapolation at an ASSUMED effective T4 throughput, to be replaced
by a 5-minute timing smoke on Colab before the epoch count is registered.

Candidate: a DDPM-style U-Net at 32 x 32 (MNIST padded from 28), base width C, multipliers
(1, 2, 2), two residual blocks per level, self-attention at 8 x 8 and in the mid-block. The
clock is a sinusoidal embedding through an MLP, added inside every residual block; "clock off"
drops it, "clock dropout" swaps in a learned null vector on a fraction of the batch.

Run:  ~/venvs/semcore/bin/python windows/power/w3_budget_check.py
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

HERE = os.path.dirname(os.path.abspath(__file__))
T4_EFFECTIVE_TFLOPS = (8.0, 15.0)     # assumption: fp16 autocast, small convs; T4 peak is 65
BATCH = 128


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


def main():
    torch.manual_seed(0)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log("W3 budget check — §4 candidate sizes and T4 time (extrapolated; Colab timing smoke decides)")
    log(f"batch {BATCH}; assumed effective T4 throughput {T4_EFFECTIVE_TFLOPS} TFLOP/s")
    for cin, name in ((1, "MNIST 32x32"), (3, "CIFAR-10 32x32")):
        for C in (32, 48, 64):
            m = UNet(cin=cin, C=C)
            params = sum(p.numel() for p in m.parameters())
            x = torch.randn(8, cin, 32, 32)
            t = torch.randint(0, 1000, (8,))
            with FlopCounterMode(display=False) as fc:
                loss = F.mse_loss(m(x, t), x)
                loss.backward()
            per_img = fc.get_total_flops() / 8
            step = per_img * BATCH
            ep = 60000 if cin == 1 else 50000
            steps_per_epoch = ep / BATCH
            lo, hi = (step / (tf * 1e12) for tf in T4_EFFECTIVE_TFLOPS[::-1])
            ep_lo, ep_hi = (3600 / (s * steps_per_epoch) for s in (hi, lo))
            log(f"  {name:15s} C {C:2d}: {params / 1e6:5.1f} M params, {per_img / 1e9:5.2f} GFLOP per "
                f"image (fwd+bwd); one step {lo:.3f}-{hi:.3f} s ⇒ {ep_lo:4.0f}-{ep_hi:4.0f} epochs per T4 hour")
            # sampling + activation harvest cost: forward only, ~1/3 of fwd+bwd
            fwd = per_img / 3
            n_evals = 4 * 3 * 256 * 2 + 3000          # 2x2 reads, 3 offsets, 256 images, both directions, + calibration
            log(f"      arm reads: {n_evals} forward passes ≈ {n_evals * fwd / (T4_EFFECTIVE_TFLOPS[0] * 1e12):.1f} s")
    with open(os.path.join(HERE, "w3_budget_check_output.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
