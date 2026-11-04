"""Vehicle and tyre parameters.

The default set is CommonRoad's vehicle 2 (BMW 320i; Althoff & Würsching, "CommonRoad: Vehicle Models",
2020, Tab. 6/7; commonroad-vehicle-models, BSD licence), so the simulator can be checked against the
CommonRoad multi-body model with identical masses, geometry, suspension and tyres.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class TireParams:
    # longitudinal
    p_cx1: float = 1.6411
    p_dx1: float = 1.1739
    p_dx3: float = 0.0
    p_ex1: float = 0.46403
    p_kx1: float = 22.303
    p_hx1: float = 0.0012297
    p_vx1: float = -8.8098e-06
    r_bx1: float = 13.276
    r_bx2: float = -13.778
    r_cx1: float = 1.2568
    r_ex1: float = 0.65225
    r_hx1: float = 0.0050722
    # lateral
    p_cy1: float = 1.3507
    p_dy1: float = 1.0489
    p_dy3: float = -2.8821
    p_ey1: float = -0.0074722
    p_ky1: float = -21.92
    p_hy1: float = 0.0026747
    p_hy3: float = 0.031415
    p_vy1: float = 0.037318
    p_vy3: float = -0.32931
    r_by1: float = 7.1433
    r_by2: float = 9.1916
    r_by3: float = -0.027856
    r_cy1: float = 1.0719
    r_ey1: float = -0.27572
    r_hy1: float = 5.7448e-06
    r_vy1: float = -0.027825
    r_vy3: float = -0.27568
    r_vy4: float = 12.12
    r_vy5: float = 1.9
    r_vy6: float = -10.704


@dataclass
class VehicleParams:
    m: float = 1093.2952334674046  # total mass [kg]
    m_s: float = 965.7108098804363  # sprung mass
    m_uf: float = 63.7921826056784  # unsprung mass front axle
    m_ur: float = 63.7921826056784  # unsprung mass rear axle
    a: float = 1.1561957064  # CoG to front axle [m]
    b: float = 1.4227170936  # CoG to rear axle [m]
    h_s: float = 0.61373004  # sprung-mass CoG height [m]
    I_z: float = 1791.5995300122856  # yaw inertia [kg m^2]
    I_phi_s: float = 207.26524557936952  # sprung-mass roll inertia
    I_y_s: float = 1565.8178787125541  # sprung-mass pitch inertia
    T_f: float = 1.38684  # track width front [m]
    T_r: float = 1.36398  # track width rear [m]
    K_sf: float = 24453.137879749014  # suspension spring rate front [N/m] (per wheel)
    K_sr: float = 19635.504745231297
    K_sdf: float = 1786.2441002440723  # suspension damping front [N s/m]
    K_sdr: float = 1649.0833034887382
    K_tsf: float = -6914.881688272133  # auxiliary torsion roll stiffness front [N m/rad] (adds stiffness, see dynamics)
    K_tsr: float = -2643.6009520155308
    K_zt: float = 158294.1398119115  # tyre vertical stiffness [N/m]
    D_f: float = -0.39370078740157477  # camber change per suspension travel front [rad/m]
    D_r: float = -0.905511811023622
    R_w: float = 0.344  # effective wheel radius [m]
    I_y_w: float = 1.7  # wheel spin inertia [kg m^2]
    T_sb: float = 0.66  # brake torque share of the front axle
    T_se: float = 0.0  # engine torque share of the front axle (0: rear-wheel drive)
    # input constraints (CommonRoad steering / longitudinal)
    steer_max: float = 1.066  # [rad]
    steer_rate_max: float = 0.4  # [rad/s]
    a_max: float = 11.5  # [m/s^2]
    v_switch: float = 7.319  # above it the drive limit falls as a_max * v_switch / v
    v_max: float = 50.8
    tire: TireParams = field(default_factory=TireParams)

    @property
    def L(self) -> float:
        return self.a + self.b


def from_commonroad(p) -> VehicleParams:
    """A ``VehicleParams`` from a CommonRoad parameter object (``vehiclemodels.parameters_vehicleN()``)."""
    tire = TireParams(**{f.name: float(getattr(p.tire, f.name)) for f in fields(TireParams)})
    names = {f.name for f in fields(VehicleParams)} - {"tire", "I_phi_s", "steer_max", "steer_rate_max", "a_max", "v_switch", "v_max"}
    kw = {n: float(getattr(p, n)) for n in names}
    return VehicleParams(
        **kw, I_phi_s=float(p.I_Phi_s), steer_max=float(p.steering.max), steer_rate_max=float(p.steering.v_max),
        a_max=float(p.longitudinal.a_max), v_switch=float(p.longitudinal.v_switch), v_max=float(p.longitudinal.v_max), tire=tire,
    )
