"""PAC2002-type magic formula, batched: the CommonRoad tyre equations (``vehiclemodels.utils.tire_model``)
term by term, on tensors, with an extra road-friction scale ``mu`` on the peak coefficients (1 = the
tyre's own dry-road peak) and a slip-stiffness scale ``stiffness`` (1 = nominal). Turn slip neglected, all other scaling factors 1, as in CommonRoad.

Sign conventions follow CommonRoad: longitudinal slip ``s = 1 - R omega / u`` (positive when braking),
slip angle ``alpha`` positive when the contact patch moves to the left of the wheel heading.
"""

from __future__ import annotations

import torch

from supermimic.vehicles.params import TireParams


def _mf(B, C, D, E, x):
    bx = B * x
    return D * torch.sin(C * torch.atan(bx - E * (bx - torch.atan(bx))))


def longitudinal(kappa, gamma, fz, p: TireParams, mu=1.0, stiffness=1.0):
    kappa = -kappa
    kx = kappa + p.p_hx1
    mu_x = mu * p.p_dx1 * (1 - p.p_dx3 * gamma**2)
    D = mu_x * fz
    B = stiffness * fz * p.p_kx1 / (p.p_cx1 * D)
    bx = B * kx
    # CommonRoad adds the vertical shift S_vx inside the sine (kept for parity; S_vx is ~1e-5 Fz).
    return D * torch.sin(p.p_cx1 * torch.atan(bx - p.p_ex1 * (bx - torch.atan(bx))) + fz * p.p_vx1)


def lateral(alpha, gamma, fz, p: TireParams, mu=1.0, stiffness=1.0):
    sg = torch.sign(gamma)
    shy = sg * (p.p_hy1 + p.p_hy3 * gamma.abs())
    svy = sg * fz * (p.p_vy1 + p.p_vy3 * gamma.abs())
    mu_y = mu * p.p_dy1 * (1 - p.p_dy3 * gamma**2)
    D = mu_y * fz
    B = stiffness * fz * p.p_ky1 / (p.p_cy1 * D)
    return _mf(B, p.p_cy1, D, p.p_ey1, alpha + shy) + svy, mu_y


def _cos_mf(B, C, E, x):
    bx = B * x
    return torch.cos(C * torch.atan(bx - E * (bx - torch.atan(bx))))


def longitudinal_comb(kappa, alpha, fx0, p: TireParams):
    B = p.r_bx1 * torch.cos(torch.atan(p.r_bx2 * kappa))
    D = fx0 / _cos_mf(B, p.r_cx1, p.r_ex1, torch.full_like(kappa, p.r_hx1))
    return D * _cos_mf(B, p.r_cx1, p.r_ex1, alpha + p.r_hx1)


def lateral_comb(kappa, alpha, gamma, mu_y, fz, fy0, p: TireParams):
    B = p.r_by1 * torch.cos(torch.atan(p.r_by2 * (alpha - p.r_by3)))
    D = fy0 / _cos_mf(B, p.r_cy1, p.r_ey1, torch.full_like(kappa, p.r_hy1))
    dv = mu_y * fz * (p.r_vy1 + p.r_vy3 * gamma) * torch.cos(torch.atan(p.r_vy4 * alpha))
    sv = dv * torch.sin(p.r_vy5 * torch.atan(p.r_vy6 * kappa))
    return D * _cos_mf(B, p.r_cy1, p.r_ey1, kappa + p.r_hy1) + sv


def forces(kappa, alpha, gamma, fz, p: TireParams, mu=1.0, stiffness=1.0):
    """Combined-slip (Fx, Fy) per wheel, as CommonRoad's multi-body model computes them."""
    fx0 = longitudinal(kappa, gamma, fz, p, mu, stiffness)
    fy0, mu_y = lateral(alpha, gamma, fz, p, mu, stiffness)
    return longitudinal_comb(kappa, alpha, fx0, p), lateral_comb(kappa, alpha, gamma, mu_y, fz, fy0, p)
