#!/usr/bin/env python3
"""Numerical check of explicit-key and absorbed-query GVA score formulations."""
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)

b, h, g, d, r, t = 1, 14, 2, 64, 16, 512
dn = d - r

def rand(shape):
    return torch.randn(shape, device=device, dtype=torch.float32)

qn = rand((b, h, dn))
qr = rand((b, h, r))
vc = rand((b, g, t, d))
kr = rand((b, t, r))
maps = rand((h, dn, d)) / (dn ** 0.5)

mapping = torch.arange(h, device=device) // (h // g)
v_per_head = vc[:, mapping, :, :]

q_absorbed = torch.einsum("bhn,hnd->bhd", qn, maps)
absorbed_content = torch.einsum("bhd,bhtd->bht", q_absorbed, v_per_head)

k_explicit = torch.einsum("bhtd,hnd->bhtn", v_per_head, maps)
explicit_content = torch.einsum("bhn,bhtn->bht", qn, k_explicit)

positional = torch.einsum("bhr,btr->bht", qr, kr)
scale = (dn + r) ** -0.5
absorbed_scores = (absorbed_content + positional) * scale
explicit_scores = (explicit_content + positional) * scale

def report(label, a, b):
    err = (a - b).abs().max().item()
    passed = torch.allclose(a, b, rtol=1e-5, atol=1e-5)
    print(f"{label}: max_abs_error={err:.9e}, allclose={passed}")
    if not passed:
        raise SystemExit(f"{label} failed")

report("Content scores", absorbed_content, explicit_content)
report("Full attention scores", absorbed_scores, explicit_scores)
print(f"device={device}, dtype=float32, d_n={dn}, d_r={r}, d={d}")
