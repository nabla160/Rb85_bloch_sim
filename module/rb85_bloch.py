
import numpy as np
import scipy.constants as sc
from sympy.physics.wigner import clebsch_gordan

# ---------------------------------------------------------------------------
# Atomic basis: 85Rb D2, F=3 -> F'=4 
# ---------------------------------------------------------------------------
FG = 3
FE = 4

MF3 = list(range(-FG, FG + 1))            # ground mF, -3..3 (7 states)
MFP4 = list(range(-FE, FE + 1))           # excited mF, -4..4 (9 states)

MF3_INDEX = {m: i for i, m in enumerate(MF3)}
MFP4_INDEX = {m: len(MF3) + i for i, m in enumerate(MFP4)}

N_G = len(MF3)            # 7
N_E = len(MFP4)           # 9
N_CLOSED = N_G + N_E       # 16
Q_LIST = [-1, 0, 1]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GAMMA = 2 * np.pi * 6.065e6                        # rad/s

S_level_to_P_level = 384.230406373e6   # MHz, Steck Rb85 D Line Data
S_level_to_F3 = 1264.8885163           # MHz
P_level_to_F4 = 100.357                # MHz
SF3_toPF4 = S_level_to_P_level - S_level_to_F3 + P_level_to_F4   # MHz

OMEGA0 = 2 * np.pi * SF3_toPF4 * 1e6                # rad/s, transition frequency
K0 = OMEGA0 / sc.c                                   # rad/m, vacuum wave vector
I_SAT = (sc.hbar * GAMMA * OMEGA0 ** 3) / (12 * np.pi * sc.c ** 2)   # W/m^2
MU_B = (sc.e * sc.hbar) / (2 * sc.m_e)               # J/T, Bohr magneton

G_GROUND = 1 / 3     # 85Rb 5^2S_1/2 F=3 Lande g-factor 
G_EXCITED = 1 / 2    # 85Rb 5^2P_3/2 F'=4 Lande g-factor


# ---------------------------------------------------------------------------
# Clebsch-Gordan coefficients
# ---------------------------------------------------------------------------
def cg_coefficient(mg, q, me):
    return complex(clebsch_gordan(FG, 1, FE, mg, q, me).evalf())


# ---------------------------------------------------------------------------
# Dipole coupling operators 
# ---------------------------------------------------------------------------
def dipole_coupling_operator(q):
    V = np.zeros((N_CLOSED, N_CLOSED), dtype=complex)
    for mg in MF3:
        me = mg + q
        if me not in MFP4_INDEX:
            continue
        V[MFP4_INDEX[me], MF3_INDEX[mg]] = cg_coefficient(mg, q, me)
    return V


_V_OPS = {q: dipole_coupling_operator(q) for q in Q_LIST}


# ---------------------------------------------------------------------------
# Angular-momentum operators Fx, Fy, Fz 
# ---------------------------------------------------------------------------
def angular_momentum_ops(F):
    """Fx, Fy, Fz as (2F+1)x(2F+1) matrices in the |F,mF> basis, mF=-F..F
    ascending -- standard ladder-operator construction, same convention
    (and same function name/signature) as the reference codebase's
    module.rb85_bloch.angular_momentum_ops."""
    dim = int(round(2 * F + 1))
    mFs = np.arange(-F, F + 0.5, 1.0)
    Fz = np.diag(mFs).astype(complex)
    Fp_op = np.zeros((dim, dim), dtype=complex)
    for i, mF in enumerate(mFs[:-1]):
        Fp_op[i + 1, i] = np.sqrt(F * (F + 1) - mF * (mF + 1))
    Fm_op = Fp_op.conj().T
    Fx = (Fp_op + Fm_op) / 2
    Fy = (Fp_op - Fm_op) / (2j)
    return Fx, Fy, Fz


_FX_G, _FY_G, _FZ_G = angular_momentum_ops(FG)
_FX_E, _FY_E, _FZ_E = angular_momentum_ops(FE)


# ---------------------------------------------------------------------------
# Hamiltonian pieces
# ---------------------------------------------------------------------------
def zeeman_hamiltonian(B_vec):
    
    Bx, By, Bz = B_vec
    H = np.zeros((N_CLOSED, N_CLOSED), dtype=complex)
    H[:N_G, :N_G] = G_GROUND * MU_B / sc.hbar * (Bx * _FX_G + By * _FY_G + Bz * _FZ_G)
    H[N_G:, N_G:] = G_EXCITED * MU_B / sc.hbar * (Bx * _FX_E + By * _FY_E + Bz * _FZ_E)
    return H


# Named transverse Jones vectors in the (e1, e2) basis
PUMP_POLARIZATIONS = {
    'H': np.array([1.0, 0.0], dtype=complex),
    'V': np.array([0.0, 1.0], dtype=complex),
    'D': np.array([1.0, 1.0], dtype=complex) / np.sqrt(2),
    'A': np.array([1.0, -1.0], dtype=complex) / np.sqrt(2),
    'R': np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2),
    'L': np.array([1.0, -1.0j], dtype=complex) / np.sqrt(2),
}


def pump_spherical_components(pol):
    if isinstance(pol, str):
        pol = PUMP_POLARIZATIONS[pol]
    E = np.asarray(pol, dtype=complex)
    E = E / np.linalg.norm(E)
    return U.conj().T @ np.array([E[0], E[1], 0.0])


def pump_hamiltonian(pump_s, q=-1, pol=None):
    Omega_0 = GAMMA * np.sqrt(max(pump_s, 0.0) / 2)
    if pol is None:
        V_pump = _V_OPS[q]
    else:
        c = pump_spherical_components(pol)
        V_pump = sum(c[i] * _V_OPS[qi] for i, qi in enumerate(Q_LIST))
    H = np.zeros((N_CLOSED, N_CLOSED), dtype=complex)
    H += (Omega_0 / 2) * V_pump + (Omega_0 / 2) * V_pump.conj().T
    return H


# ---------------------------------------------------------------------------
# Spontaneous-emission jump operators 
# ---------------------------------------------------------------------------
def _spontaneous_jump_operators():
    jumps = []
    for me in MFP4:
        branching_sum = 0.0
        for q in Q_LIST:
            mg = me - q
            if mg not in MF3_INDEX:
                continue
            ratio = abs(cg_coefficient(mg, q, me)) ** 2
            if ratio < 1e-15:
                continue
            C = np.zeros((N_CLOSED, N_CLOSED), dtype=complex)
            C[MF3_INDEX[mg], MFP4_INDEX[me]] = np.sqrt(GAMMA * ratio)
            jumps.append(C)
            branching_sum += ratio
        assert np.isclose(branching_sum, 1.0, atol=1e-12), \
            f"Invalid branching sum for mF'={me}: {branching_sum}"
    return jumps


_JUMPS = _spontaneous_jump_operators()


# ---------------------------------------------------------------------------
# Liouvillian
# ---------------------------------------------------------------------------
def build_liouvillian_coherent(H):
    """Vectorized -i[H,rho] superoperator (N_CLOSED^2 x N_CLOSED^2). C-order
    vec(rho)=rho.reshape(-1) satisfies vec(A rho B) = (A kron B^T) vec(rho)."""
    I = np.eye(N_CLOSED, dtype=complex)
    return -1j * (np.kron(H, I) - np.kron(I, H.T))


def build_liouvillian_dissipator(jumps):
    """Sum of Lindblad dissipator superoperators C rho C^dagger -
    1/2{C^dagger C, rho} for each spontaneous-decay jump operator."""
    D = np.zeros((N_CLOSED * N_CLOSED, N_CLOSED * N_CLOSED), dtype=complex)
    I = np.eye(N_CLOSED, dtype=complex)
    for C in jumps:
        Cd = C.conj().T
        CdC = Cd @ C
        D += np.kron(C, C.conj()) - 0.5 * (np.kron(CdC, I) + np.kron(I, CdC.T))
    return D


# ---------------------------------------------------------------------------
# Steady-state solution
# ---------------------------------------------------------------------------
def steady_state_rho_closed(B_vec, pump_s, pump_detuning=0.0, gamma_t=0.0, rho_in=None,
                            pump_pol=None):
    B_vec = np.asarray(B_vec, dtype=float)
    H = zeeman_hamiltonian(B_vec) + pump_hamiltonian(pump_s, pol=pump_pol)
    H[N_G:, N_G:] += -pump_detuning * np.eye(N_E)

    L = (build_liouvillian_coherent(H)
         + build_liouvillian_dissipator(_JUMPS)
         - gamma_t * np.eye(N_CLOSED * N_CLOSED, dtype=complex))

    if rho_in is None:
        rho_in = np.zeros((N_CLOSED, N_CLOSED), dtype=complex)
        for m in MF3:
            rho_in[MF3_INDEX[m], MF3_INDEX[m]] = 1.0 / N_G

    if gamma_t > 0:
        rhs = -gamma_t * rho_in.reshape(-1)
        rho = np.linalg.solve(L, rhs).reshape(N_CLOSED, N_CLOSED)
    else:
        trace_row = np.zeros(N_CLOSED * N_CLOSED, dtype=complex)
        for i in range(N_CLOSED):
            trace_row[i * N_CLOSED + i] = 1.0
        A = L.copy()
        b = np.zeros(N_CLOSED * N_CLOSED, dtype=complex)
        A[0, :] = trace_row
        b[0] = 1.0
        rho = np.linalg.solve(A, b).reshape(N_CLOSED, N_CLOSED)

    rho = (rho + rho.conj().T) / 2   # enforce Hermiticity against solver round-off

    min_eig = np.linalg.eigvalsh(rho).min()
    if min_eig < -1e-9:
        raise ValueError(f"Non-physical steady state: smallest eigenvalue {min_eig:.3e}")

    return rho, L


def diagnostics(rho, L, gamma_t, rho_in=None):
    if rho_in is None:
        rho_in = np.zeros((N_CLOSED, N_CLOSED), dtype=complex)
        for m in MF3:
            rho_in[MF3_INDEX[m], MF3_INDEX[m]] = 1.0 / N_G

    drho = (L @ rho.reshape(-1)).reshape(N_CLOSED, N_CLOSED) + gamma_t * rho_in
    residual = np.linalg.norm(drho)

    return {
        "trace": np.trace(rho),
        "hermiticity_error": np.linalg.norm(rho - rho.conj().T),
        "absolute_residual": residual,
        "relative_residual": residual / (GAMMA * np.linalg.norm(rho)),
        "min_eigenvalue": np.linalg.eigvalsh((rho + rho.conj().T) / 2).min(),
    }


# ---------------------------------------------------------------------------
# Weak-probe linear response -> susceptibility tensor
# ---------------------------------------------------------------------------
def probe_linear_response(delta_probe, V_probe, rho_ss, L):
    n2 = L.shape[0]
    commutator = V_probe @ rho_ss - rho_ss @ V_probe
    rhs = 1j * 0.5 * commutator.reshape(-1)
    A = L + 1j * delta_probe * np.eye(n2, dtype=complex)
    delta_rho = np.linalg.solve(A, rhs)
    return delta_rho.reshape((N_CLOSED, N_CLOSED))


def chi_tensor_from_rho(detuning, rho_ss, L, n_rho):
    detuning = np.atleast_1d(np.asarray(detuning, dtype=float))
    N = len(detuning)
    chi = np.zeros((N, 3, 3), dtype=complex)

    prefac = -(n_rho * sc.hbar * sc.c) / 2 * (GAMMA ** 2 / I_SAT)

    for qpi, qp in enumerate(Q_LIST):
        V_qp = _V_OPS[qp]
        for i, delta in enumerate(detuning):
            delta_rho = probe_linear_response(delta, V_qp, rho_ss, L)
            for qi, q in enumerate(Q_LIST):
                V_q = _V_OPS[q]
                chi[i, qi, qpi] = prefac * np.trace(delta_rho @ (V_q + V_q.conj().T))

    return chi


# ---------------------------------------------------------------------------
# Doppler averaging (test)
MASS_RB85 = 84.911789738 * sc.physical_constants['atomic mass constant'][0]   # kg


def maxwell_boltzmann_velocity_grid(T_kelvin, n_v=201, v_range_factor=5.0, mass=MASS_RB85,
                                     concentration=0.08):
    u = np.sqrt(2 * sc.k * T_kelvin / mass)
    if not concentration or concentration <= 0:
        v_grid = np.linspace(-v_range_factor * u, v_range_factor * u, n_v)
    else:
        edge = np.arctan(v_range_factor / concentration)
        t = np.linspace(-edge, edge, n_v)
        v_grid = concentration * u * np.tan(t)

    dv_local = np.gradient(v_grid)
    unnorm = np.exp(-(v_grid / u) ** 2) * dv_local
    weights = unnorm / np.sum(unnorm)
    return v_grid, weights


def steady_state_rho_doppler_v(B_vec, pump_s, v, pump_detuning=0.0, gamma_t=0.0,
                                pump_propagation_sign=-1, rho_in=None):
    """steady_state_rho_closed for a single velocity class v (m/s along z),
    with the PUMP's own Doppler-shifted detuning applied -- see module note
    above. pump_detuning here is the LAB-frame pump detuning (what you'd
    dial in on the laser), not yet Doppler-shifted."""
    pump_detuning_eff = pump_detuning - pump_propagation_sign * K0 * v
    return steady_state_rho_closed(B_vec, pump_s, pump_detuning_eff, gamma_t, rho_in)


def chi_tensor_doppler_averaged(detuning, B_vec, pump_s, T_kelvin, n_rho,
                                 pump_detuning=0.0, gamma_t=0.0, pump_propagation_sign=-1,
                                 n_v=201, v_range_factor=5.0, concentration=0.08):
    detuning = np.atleast_1d(np.asarray(detuning, dtype=float))
    v_grid, weights = maxwell_boltzmann_velocity_grid(T_kelvin, n_v, v_range_factor, concentration=concentration)

    chi_avg = np.zeros((len(detuning), 3, 3), dtype=complex)
    for v, w in zip(v_grid, weights):
        rho_v, L_v = steady_state_rho_doppler_v(
            B_vec, pump_s, v, pump_detuning, gamma_t, pump_propagation_sign
        )
        delta_eff = detuning - K0 * v * (1 - pump_propagation_sign)
        chi_v = chi_tensor_from_rho(delta_eff, rho_v, L_v, n_rho)
        chi_avg += chi_v * w

    return chi_avg


U = np.array(
    [[(-1 / np.sqrt(2)), 0, (1 / np.sqrt(2))],
     [(-1j / np.sqrt(2)), 0, (-1j / np.sqrt(2))],
     [0, 1, 0]]
)   # spherical (q=-1,0,+1) -> Cartesian (e1,e2,k_hat), same convention as rb85_cell.U


def chi_spherical_to_cartesian(chi_q):
    
    return np.array([U @ chi_n @ U.conj().T for chi_n in chi_q])


def transverse_basis(k_hat):
   
    k_hat = np.asarray(k_hat, dtype=float)
    k_hat = k_hat / np.linalg.norm(k_hat)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, k_hat)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(helper, k_hat)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(k_hat, e1)
    return e1, e2, k_hat


def jones_from_chi_cartesian(chi_cart, L, schur=True):
   
    J = np.zeros((len(chi_cart), 2, 2), dtype=complex)
    for n, chi_n in enumerate(chi_cart):
        eps = np.eye(3, dtype=complex) + chi_n
        if schur:
            M_eff = eps[:2, :2] - np.outer(eps[:2, 2], eps[2, :2]) / eps[2, 2]
        else:
            M_eff = eps[:2, :2]
        vp, V = np.linalg.eig(M_eff)
        P = np.diag(np.exp(1j * K0 * np.sqrt(vp) * L))
        J[n] = V @ P @ np.linalg.inv(V)
    return J


def jones_matrix_closed(detuning, B_vec, pump_s, L_cell, n_rho,
                         pump_detuning=0.0, gamma_t=0.0, schur=True,
                         pump_pol=None):
  
    detuning = np.atleast_1d(np.asarray(detuning, dtype=float))
    rho_ss, L = steady_state_rho_closed(B_vec, pump_s, pump_detuning, gamma_t,
                                        pump_pol=pump_pol)
    chi_q = chi_tensor_from_rho(detuning, rho_ss, L, n_rho)
    chi_cart = chi_spherical_to_cartesian(chi_q)
    return jones_from_chi_cartesian(chi_cart, L_cell, schur=schur)


def jones_matrix_approx(detuning, B_vec, pump_s, L_cell, n_rho,
                         pump_detuning=0.0, gamma_t=0.0, pump_pol=None):
    
    return jones_matrix_closed(detuning, B_vec, pump_s, L_cell, n_rho,
                                pump_detuning, gamma_t, schur=False,
                                pump_pol=pump_pol)


def propagate_closed(E_in, detuning, B_vec, pump_s, L_cell, n_rho,
                      pump_detuning=0.0, gamma_t=0.0, schur=True):
   
    E_in = np.asarray(E_in, dtype=complex)
    J = jones_matrix_closed(detuning, B_vec, pump_s, L_cell, n_rho,
                             pump_detuning, gamma_t, schur=schur)
    return np.array([Jn @ E_in for Jn in J])


def propagate_approx(E_in, detuning, B_vec, pump_s, L_cell, n_rho,
                      pump_detuning=0.0, gamma_t=0.0):
   
    E_in = np.asarray(E_in, dtype=complex)
    J = jones_matrix_approx(detuning, B_vec, pump_s, L_cell, n_rho, pump_detuning, gamma_t)
    return np.array([Jn @ E_in for Jn in J])


def jones_matrix_doppler(detuning, B_vec, pump_s, L_cell, n_rho, T_kelvin,
                          pump_detuning=0.0, gamma_t=0.0, pump_propagation_sign=-1,
                          n_v=201, v_range_factor=5.0, concentration=0.08, schur=True):
   
    detuning = np.atleast_1d(np.asarray(detuning, dtype=float))
    chi_q = chi_tensor_doppler_averaged(
        detuning, B_vec, pump_s, T_kelvin, n_rho, pump_detuning, gamma_t,
        pump_propagation_sign, n_v, v_range_factor, concentration,
    )
    chi_cart = chi_spherical_to_cartesian(chi_q)
    return jones_from_chi_cartesian(chi_cart, L_cell, schur=schur)


def propagate_doppler(E_in, detuning, B_vec, pump_s, L_cell, n_rho, T_kelvin,
                       pump_detuning=0.0, gamma_t=0.0, pump_propagation_sign=-1,
                       n_v=201, v_range_factor=5.0, concentration=0.08, schur=True):
   
    E_in = np.asarray(E_in, dtype=complex)
    J = jones_matrix_doppler(
        detuning, B_vec, pump_s, L_cell, n_rho, T_kelvin, pump_detuning, gamma_t,
        pump_propagation_sign, n_v, v_range_factor, concentration, schur=schur,
    )
    return np.array([Jn @ E_in for Jn in J])


def stokes(E_out):
   
    Ex = E_out[..., 0]
    Ey = E_out[..., 1]
    S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = -2 * np.imag(Ex * np.conj(Ey))
    return np.stack([S0, S1, S2, S3], axis=-1)


def polarization_intensities(E_out):
   
    S = stokes(E_out)
    S0, S1, S2, S3 = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    return {
        'H': (S0 + S1) / 2,
        'V': (S0 - S1) / 2,
        'D': (S0 + S2) / 2,
        'A': (S0 - S2) / 2,
        'R': (S0 + S3) / 2,
        'L': (S0 - S3) / 2,
    }


def polarization_angle(E_out):
   
    S = stokes(E_out)
    return 0.5 * np.arctan2(S[..., 2], S[..., 1])
