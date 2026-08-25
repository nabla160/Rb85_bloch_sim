
from dataclasses import dataclass

import numpy as np

import module.rb85_bloch as rb85_bloch

N_G = rb85_bloch.N_G
MF3 = rb85_bloch.MF3
MFP4 = rb85_bloch.MFP4

theta_values = np.linspace(0, 2 * np.pi, 121)
_design = np.column_stack([np.ones_like(theta_values),
                           np.sin(2 * theta_values),
                           np.cos(2 * theta_values)])
_design2 = np.column_stack([np.sin(2 * theta_values), np.cos(2 * theta_values)])

COMPONENT_STYLES = [
    ('H', '$I_H$', 'tab:blue',   '-'),
    ('V', '$I_V$', 'tab:orange', '-'),
    ('R', '$I_R$', 'tab:green',  '--'),
    ('L', '$I_L$', 'tab:red',    '--'),
    ('D', '$I_D$', 'tab:purple', ':'),
    ('A', '$I_A$', 'tab:brown',  ':'),
]
AMP_THRESHOLD = 1e-2


@dataclass(frozen=True)
class Params:
    pump_pol: str = 'R'
    det_scan_mode: str = 'probe'   # 'probe' | 'both' | 'pump'
    pump_s: float = 0.6
    pump_det_MHz: float = 0.0
    gamma_t: float = 3.0e4
    n_rho: float = 5.383e13
    L_cell_mm: float = 75.0
    B_norm_G: float = 0.5
    theta_B_deg: float = 0.0
    phi_B_deg: float = 0.0
    cut_theta_deg: float = 0.0
    cut_phi_deg: float = 90.0
    det_max_MHz: float = 20.0
    B_max_G: float = 1.0
    det_fixed_MHz: float = 0.0
    n_det: int = 41
    n_B: int = 25


@dataclass
class Results:
    B_dir: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    populations: np.ndarray
    det_MHz: np.ndarray
    ph_det: dict
    am_det: dict
    B_scan_G: np.ndarray
    ph_B: dict
    am_B: dict


def _fit_phase(y):
    """Return (phase, amplitude) of the sin(2*theta+phi) fit of y(theta)."""
    c, a, b = np.linalg.lstsq(_design, y, rcond=None)[0]
    A = np.sqrt(a**2 + b**2)
    if A < 1e-30:
        return np.nan, A
    sin_c, cos_c = np.linalg.lstsq(_design2, (y - c) / A, rcond=None)[0]
    return np.arctan2(cos_c, sin_c), A


def _zero_at_zero(x, phase_arr):
    p = np.unwrap(np.where(np.isfinite(phase_arr), phase_arr, np.nan))
    finite = np.isfinite(p)
    if not np.any(finite):
        return p
    return p - np.interp(0, x[finite], p[finite])


def wrap_pi(phase):
    return (phase + np.pi) % (2 * np.pi) - np.pi


def make_cut(cut_theta, cut_phi):
    """Basis (e1, e2) of the great circle with normal (cut_theta, cut_phi)."""
    n = np.array([np.sin(cut_phi) * np.cos(cut_theta),
                  np.sin(cut_phi) * np.sin(cut_theta),
                  np.cos(cut_phi)])
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, n)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def E_in_on_cut(e1, e2):
    """Poincare -> Jones conversion along the cut: complex (n_theta, 2)."""
    E = np.empty((len(theta_values), 2), dtype=complex)
    for i, th in enumerate(theta_values):
        s1, s2, s3 = np.cos(2 * th) * e1 + np.sin(2 * th) * e2
        alpha = np.arccos(np.clip(s1, -1.0, 1.0)) / 2
        delta = np.arctan2(s3, s2)
        E[i] = [np.cos(alpha), np.sin(alpha) * np.exp(1j * delta)]
    return E


def _fit_scan(I, x):
    phases, amps = {}, {}
    for key, *_ in COMPONENT_STYLES:
        pa = np.array([_fit_phase(I[key][:, j]) for j in range(I[key].shape[1])])
        phases[key] = _zero_at_zero(x, pa[:, 0])
        amps[key] = pa[:, 1]
    return phases, amps


def compute(p: Params) -> Results:
    th, ph = np.radians(p.theta_B_deg), np.radians(p.phi_B_deg)
    B_dir = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    B_T = p.B_norm_G * 1e-4 * B_dir
    pump_det_rad = 2 * np.pi * p.pump_det_MHz * 1e6
    L_cell = p.L_cell_mm * 1e-3

    rho_ss, _ = rb85_bloch.steady_state_rho_closed(B_T, p.pump_s, pump_det_rad, p.gamma_t,
                                                   pump_pol=p.pump_pol)
    populations = np.real(np.diag(rho_ss))

    e1, e2 = make_cut(np.radians(p.cut_theta_deg), np.radians(p.cut_phi_deg))
    E_in = E_in_on_cut(e1, e2)

    def jones_at(beat_rad, pump_det_rad_eff, B_vec_T):
        return rb85_bloch.jones_matrix_closed(
            np.atleast_1d(beat_rad), B_vec_T, p.pump_s, L_cell, p.n_rho,
            pump_detuning=pump_det_rad_eff, gamma_t=p.gamma_t, pump_pol=p.pump_pol,
        )

    # Detuning-scan convention: probe and pump both start from the base pump
    # detuning (all modes agree at scan point 0); the scanned offset D goes to
    #   'probe' : probe only          -> beat = D,  pump fixed (one rho_ss solve)
    #   'both'  : pump AND probe      -> beat = 0,  pump = base + D (solve per point)
    #   'pump'  : pump only           -> beat = -D, pump = base + D (solve per point)
    det_MHz = np.linspace(-p.det_max_MHz, p.det_max_MHz, int(round(p.n_det)))
    det_rad = 2 * np.pi * det_MHz * 1e6
    if p.det_scan_mode == 'probe':
        J_det = jones_at(det_rad, pump_det_rad, B_T)
    else:
        beat_of = (lambda d: 0.0) if p.det_scan_mode == 'both' else (lambda d: -d)
        J_det = np.array([jones_at(beat_of(d), pump_det_rad + d, B_T)[0]
                          for d in det_rad])
    I_det = rb85_bloch.polarization_intensities(np.einsum('dij,tj->tdi', J_det, E_in))

    # signed-B scan along B_dir at fixed detuning (same mode convention)
    B_scan_G = np.linspace(-p.B_max_G, p.B_max_G, int(round(p.n_B)))
    det_fixed_rad = 2 * np.pi * p.det_fixed_MHz * 1e6
    if p.det_scan_mode == 'probe':
        beat_B, pump_B = det_fixed_rad, pump_det_rad
    elif p.det_scan_mode == 'both':
        beat_B, pump_B = 0.0, pump_det_rad + det_fixed_rad
    else:
        beat_B, pump_B = -det_fixed_rad, pump_det_rad + det_fixed_rad
    J_B = np.array([
        jones_at(beat_B, pump_B, Bn * 1e-4 * B_dir)[0]
        for Bn in B_scan_G
    ])
    I_B = rb85_bloch.polarization_intensities(np.einsum('bij,tj->tbi', J_B, E_in))

    ph_det, am_det = _fit_scan(I_det, det_MHz)
    ph_B, am_B = _fit_scan(I_B, B_scan_G)

    return Results(B_dir=B_dir, e1=e1, e2=e2, populations=populations,
                   det_MHz=det_MHz, ph_det=ph_det, am_det=am_det,
                   B_scan_G=B_scan_G, ph_B=ph_B, am_B=am_B)
