import matplotlib.pyplot as plt
import numpy as np

from scipy.constants import physical_constants
from sympy.physics.wigner import clebsch_gordan


# ============================================================
# Atomic basis: 85Rb, F = 3 -> F' = 4
# ============================================================

Fg = 3
Fe = 4

ground_m = list(range(-Fg, Fg + 1))
excited_m = list(range(-Fe, Fe + 1))

ground_index = {m: i for i, m in enumerate(ground_m)}
excited_index = {m: len(ground_m) + i for i, m in enumerate(excited_m)}

dimension = len(ground_m) + len(excited_m)
liouville_dimension = dimension**2


# ============================================================
# Physical parameters
# ============================================================

Gamma = 2 * np.pi * 6.065e6
mu_B = physical_constants["Bohr magneton"][0]
hbar = physical_constants["Planck constant over 2 pi"][0]
mu_B_over_hbar = mu_B / hbar

epsilon_0 = physical_constants["vacuum electric permittivity"][0]
c_light = physical_constants["speed of light in vacuum"][0]

lambda_0 = 780.241e-9  # 85Rb D2 line, 5S1/2 -> 5P3/2 (Steck)
omega_0 = 2 * np.pi * c_light / lambda_0

# Reduced dipole matrix element, from the standard two-level relation
# Gamma = omega_0^3 d^2 / (3 pi eps0 hbar c^3), generalized to this manifold
# via the Clebsch-Gordan weighting already used for the collapse operators
# (their branching ratios sum to 1, so this is consistent for every me).
d_reduced_sq = 3 * np.pi * epsilon_0 * hbar * c_light**3 * Gamma / omega_0**3

N_density = 1e16  # atoms / m^3 -- placeholder, set to your actual beam/vapor density
L_cell = 1e-2  # probe path length through the medium, m -- placeholder, set to your actual geometry

# Magnetic field vector, in Gauss, expressed in the frame where z is the
# quantization axis (the pump-beam propagation / polarization axis).
Bx_G = 0.0
By_G = 0.0
Bz_G = 0.0

B_vec_T = np.array([Bx_G, By_G, Bz_G]) * 1e-4

I_pump = 1.65
I_sat = 1.669
saturation = I_pump / I_sat

Delta_pump = 0.0
Omega_0 = Gamma * np.sqrt(saturation / 2)

gamma_transit = 3.0e4

g_ground = 1 / 3
g_excited = 1 / 2


# ============================================================
# Clebsch-Gordan coefficients
# ============================================================

def cg_coefficient(Fg, mg, q, Fe, me):
    return complex(clebsch_gordan(Fg, 1, Fe, mg, q, me).evalf())


# ============================================================
# Dipole coupling operators (ground -> excited, per polarization q)
# ============================================================
#
# Dimensionless, CG-weighted raising operator for spherical polarization
# component q: <me| V_q |mg> = CG(Fg,mg,q,Fe,me), zero elsewhere. The pump
# Hamiltonian coupling is Omega_0/2 * V_{-1} (+ h.c.); a probe beam of
# arbitrary polarization reuses the same construction with its own q.

def dipole_coupling_operator(q):
    V = np.zeros((dimension, dimension), dtype=complex)

    for mg in ground_m:
        me = mg + q

        if me not in excited_index:
            continue

        V[excited_index[me], ground_index[mg]] = cg_coefficient(Fg, mg, q, Fe, me)

    return V


# ============================================================
# Angular-momentum operators Fx, Fy, Fz (per hyperfine manifold)
# ============================================================
#
# Zeeman splitting is diagonal in m only when B is along the quantization
# axis (z, set by the pump-beam polarization). A transverse component of B
# couples different mF within the same F, so the full vector Zeeman term
# needs Fx and Fy as well, built from the ladder operators F+/F-. Fg and Fe
# are independent hyperfine manifolds (different g-factors), so each set of
# operators is block-diagonal and built separately.

def spin_operators(m_list, index_map):
    F = max(m_list)

    Fz = np.zeros((dimension, dimension), dtype=complex)
    Fplus = np.zeros((dimension, dimension), dtype=complex)

    for m in m_list:
        Fz[index_map[m], index_map[m]] = m

        if (m + 1) in index_map:
            Fplus[index_map[m + 1], index_map[m]] = np.sqrt(
                F * (F + 1) - m * (m + 1)
            )

    Fminus = Fplus.conj().T
    Fx = (Fplus + Fminus) / 2
    Fy = (Fplus - Fminus) / (2j)

    return Fx, Fy, Fz


Fx_ground, Fy_ground, Fz_ground = spin_operators(ground_m, ground_index)
Fx_excited, Fy_excited, Fz_excited = spin_operators(excited_m, excited_index)


# ============================================================
# Hamiltonian H / hbar in the rotating frame
# ============================================================

H = np.zeros((dimension, dimension), dtype=complex)

Bx_T, By_T, Bz_T = B_vec_T

H += g_ground * mu_B_over_hbar * (
    Bx_T * Fx_ground + By_T * Fy_ground + Bz_T * Fz_ground
)
H += g_excited * mu_B_over_hbar * (
    Bx_T * Fx_excited + By_T * Fy_excited + Bz_T * Fz_excited
)

for m in excited_m:
    H[excited_index[m], excited_index[m]] += -Delta_pump

V_pump = dipole_coupling_operator(-1)

H += (Omega_0 / 2) * V_pump + (Omega_0 / 2) * V_pump.conj().T


# ============================================================
# Spontaneous-emission collapse operators
# ============================================================

collapse_operators = []

for me in excited_m:
    branching_sum = 0.0

    for q in (-1, 0, 1):
        mg = me - q

        if mg not in ground_index:
            continue

        branching_ratio = abs(cg_coefficient(Fg, mg, q, Fe, me))**2

        if branching_ratio < 1e-15:
            continue

        C = np.zeros((dimension, dimension), dtype=complex)
        C[ground_index[mg], excited_index[me]] = np.sqrt(Gamma * branching_ratio)

        collapse_operators.append(C)
        branching_sum += branching_ratio

    if not np.isclose(branching_sum, 1.0, atol=1e-12):
        raise ValueError(
            f"Invalid branching sum for me={me}: {branching_sum}"
        )


# ============================================================
# State of atoms entering the beam
# ============================================================

rho_in = np.zeros((dimension, dimension), dtype=complex)

for m in ground_m:
    rho_in[ground_index[m], ground_index[m]] = 1 / len(ground_m)


# ============================================================
# Homogeneous master equation
# ============================================================

def homogeneous_master_equation(rho):
    drho = -1j * (H @ rho - rho @ H)

    for C in collapse_operators:
        Cd = C.conj().T
        CdC = Cd @ C

        drho += C @ rho @ Cd - 0.5 * (CdC @ rho + rho @ CdC)

    drho -= gamma_transit * rho

    return drho


# ============================================================
# Liouvillian matrix
# ============================================================

def build_liouvillian():
    L = np.zeros(
        (liouville_dimension, liouville_dimension),
        dtype=complex,
    )

    for j in range(liouville_dimension):
        basis = np.zeros(liouville_dimension, dtype=complex)
        basis[j] = 1.0

        basis_matrix = basis.reshape((dimension, dimension))
        L[:, j] = homogeneous_master_equation(basis_matrix).reshape(-1)

    return L


L = build_liouvillian()


# ============================================================
# Stationary-state solution
# ============================================================

def solve_stationary_state():
    if gamma_transit > 0:
        rhs = -gamma_transit * rho_in.reshape(-1)
        rho_flat = np.linalg.solve(L, rhs)

    else:
        trace_row = np.zeros(liouville_dimension, dtype=complex)

        for i in range(dimension):
            trace_row[i * dimension + i] = 1.0

        L_constrained = L.copy()
        rhs = np.zeros(liouville_dimension, dtype=complex)

        L_constrained[0, :] = trace_row
        rhs[0] = 1.0

        rho_flat = np.linalg.solve(L_constrained, rhs)

    return rho_flat.reshape((dimension, dimension))


rho_stationary = solve_stationary_state()


# ============================================================
# Numerical checks
# ============================================================

stationary_derivative = homogeneous_master_equation(rho_stationary)
stationary_derivative += gamma_transit * rho_in

trace = np.trace(rho_stationary)

hermiticity_error = np.linalg.norm(
    rho_stationary - rho_stationary.conj().T
)

absolute_residual = np.linalg.norm(stationary_derivative)

relative_residual = (
    absolute_residual
    / (Gamma * np.linalg.norm(rho_stationary))
)

rho_hermitian = 0.5 * (
    rho_stationary + rho_stationary.conj().T
)

eigenvalues = np.linalg.eigvalsh(rho_hermitian)

print("=== Numerical checks ===")
print("Liouvillian shape:", L.shape)
print("Trace:", trace)
print("Hermiticity error:", hermiticity_error)
print("Absolute stationary residual:", absolute_residual)
print("Relative stationary residual:", relative_residual)
print("Smallest eigenvalue:", eigenvalues.min())


# ============================================================
# Populations
# ============================================================

ground_population = np.array([
    rho_stationary[ground_index[m], ground_index[m]].real
    for m in ground_m
])

excited_population = np.array([
    rho_stationary[excited_index[m], excited_index[m]].real
    for m in excited_m
])

total_ground_population = ground_population.sum()
total_excited_population = excited_population.sum()

normalized_ground_population = (
    ground_population
    / total_ground_population
)

print("\n=== Ground-state populations ===")

for m, p_abs, p_norm in zip(
    ground_m,
    ground_population,
    normalized_ground_population,
):
    print(
        f"mF = {m:+d}: "
        f"absolute = {p_abs:.6f}, "
        f"normalized in F=3 = {p_norm:.6f}"
    )

print("\n=== Excited-state populations ===")

for m, p_abs in zip(
    excited_m,
    excited_population,
):
    print(
        f"mF = {m:+d}: "
        f"absolute = {p_abs:.6f}"
    )

print("\nTotal ground population:", total_ground_population)
print("Total excited population:", total_excited_population)
print(
    "Total population:",
    total_ground_population + total_excited_population,
)


# ============================================================
# Weak-probe linear response
# ============================================================
#
# The probe is weak enough not to perturb rho_stationary, so its effect is
# obtained by linearizing the master equation around rho_stationary. Writing
# rho(t) = rho_stationary + delta_rho(t), delta_rho(t) = delta_rho_plus *
# exp(-i*delta_probe*t) + h.c. (delta_probe = probe-pump beat frequency), and
# keeping only O(Omega_probe) terms gives:
#
#   (L + i*delta_probe*Id) @ delta_rho_plus = i*(Omega_probe/2)*[V, rho_stationary]
#
# the same Liouvillian L already built for the pump, just shifted by
# i*delta_probe and driven by the commutator of rho_stationary with the
# probe's dipole coupling operator V (Omega_probe set to 1: the response is
# linear in it, so it only rescales the result, not the lineshape).

identity_liouville = np.eye(liouville_dimension, dtype=complex)


def probe_linear_response(delta_probe, V_probe):
    commutator = V_probe @ rho_stationary - rho_stationary @ V_probe
    rhs = 1j * 0.5 * commutator.reshape(-1)

    A = L + 1j * delta_probe * identity_liouville
    delta_rho_plus = np.linalg.solve(A, rhs)

    return delta_rho_plus.reshape((dimension, dimension))


# Susceptibility prefactor: P = N * d_reduced * Tr(delta_rho_plus (V+V^dagger)),
# with d_reduced^2 folded in via the Gamma <-> dipole relation above. The
# extra minus sign comes from the light-atom coupling convention Omega =
# -d.E/hbar (dropped above by fixing Omega_probe=1 with no sign), fixed here
# so that Im(chi) > 0 = absorption for this always-non-inverted medium.
susceptibility_prefactor = -3 * N_density * Gamma * lambda_0**3 / (8 * np.pi**2)


def probe_susceptibility(delta_probe, V_probe):
    delta_rho_plus = probe_linear_response(delta_probe, V_probe)
    dipole_response = np.trace(delta_rho_plus @ (V_probe + V_probe.conj().T))

    return susceptibility_prefactor * dipole_response


q_probe = -1  # probe polarization; -1 = co-polarized with the pump (transmission probe)
V_probe = dipole_coupling_operator(q_probe)

probe_detunings = np.linspace(-8 * Gamma, 8 * Gamma, 400)
chi_probe = np.array([
    probe_susceptibility(delta, V_probe) for delta in probe_detunings
])


# ============================================================
# Probe transmission
# ============================================================
#
# chi is intrinsic to the medium (per unit length); turning it into an
# observable needs the probe's actual path length through the atoms. For a
# weak probe the field envelope propagates as
#   E(z) = E(0) * exp(i*k0*(1 + chi/2)*z)
# (chi small, k0 taken at the bare atomic wavenumber since |delta| << omega_0).
# Im(chi) gives absorption (Beer-Lambert law); Re(chi) a dispersive phase shift.

k0 = omega_0 / c_light

optical_depth = k0 * chi_probe.imag * L_cell
probe_transmission = np.exp(-optical_depth)
probe_phase_shift = 0.5 * k0 * chi_probe.real * L_cell


# ============================================================
# Population plot
# ============================================================

COLOR_GROUND = "#2a78d6"
COLOR_EXCITED = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_MUTED = "#898781"
COLOR_INK = "#0b0b0b"
COLOR_SURFACE = "#fcfcfb"

fig, (ax_ground, ax_excited) = plt.subplots(
    1, 2, figsize=(10, 4.5), sharey=True
)
fig.patch.set_facecolor(COLOR_SURFACE)

for ax, m_values, populations, title, color in (
    (ax_ground, ground_m, ground_population, "État fondamental (F = 3)", COLOR_GROUND),
    (ax_excited, excited_m, excited_population, "État excité (F' = 4)", COLOR_EXCITED),
):
    ax.set_facecolor(COLOR_SURFACE)
    ax.bar(m_values, populations, width=0.6, color=color, zorder=3)

    top_index = int(np.argmax(populations))
    ax.annotate(
        f"{populations[top_index]:.3f}",
        (m_values[top_index], populations[top_index]),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=9,
        color=COLOR_INK,
    )

    ax.set_title(title, fontsize=11, color=COLOR_INK, pad=10)
    ax.set_xlabel(r"$m_F$", fontsize=10, color=COLOR_MUTED)
    ax.set_xticks(m_values)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)

ax_ground.set_ylabel("Population", fontsize=10, color=COLOR_MUTED)

fig.suptitle(
    "Répartition de la population entre sous-niveaux Zeeman (état stationnaire)",
    fontsize=12,
    color=COLOR_INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))


# ============================================================
# Probe susceptibility plot
# ============================================================

fig2, ax_chi = plt.subplots(figsize=(7, 4.5))
fig2.patch.set_facecolor(COLOR_SURFACE)
ax_chi.set_facecolor(COLOR_SURFACE)

detuning_MHz = probe_detunings / (2 * np.pi * 1e6)

ax_chi.axhline(0, color=COLOR_GRID, linewidth=1, zorder=1)
ax_chi.plot(
    detuning_MHz, chi_probe.imag,
    color=COLOR_GROUND, linewidth=2, label="Im(χ) — absorption", zorder=3,
)
ax_chi.plot(
    detuning_MHz, chi_probe.real,
    color=COLOR_EXCITED, linewidth=2, label="Re(χ) — dispersion", zorder=3,
)

ax_chi.set_xlabel("Détuning sonde δ / 2π (MHz)", fontsize=10, color=COLOR_MUTED)
ax_chi.set_ylabel("Susceptibilité χ", fontsize=10, color=COLOR_MUTED)
ax_chi.set_title(
    f"Réponse linéaire de la sonde (q = {q_probe:+d}), "
    f"N = {N_density:.1e} m⁻³",
    fontsize=11, color=COLOR_INK, pad=10,
)

ax_chi.grid(axis="y", color=COLOR_GRID, linewidth=1, zorder=0)
ax_chi.set_axisbelow(True)

for spine in ("top", "right"):
    ax_chi.spines[spine].set_visible(False)
ax_chi.spines["left"].set_color(COLOR_AXIS)
ax_chi.spines["bottom"].set_color(COLOR_AXIS)
ax_chi.tick_params(colors=COLOR_MUTED, labelsize=9)

legend = ax_chi.legend(frameon=False, fontsize=9, loc="best")
for text in legend.get_texts():
    text.set_color(COLOR_INK)

fig2.tight_layout()


# ============================================================
# Probe transmission plot
# ============================================================

fig3, ax_T = plt.subplots(figsize=(7, 4.5))
fig3.patch.set_facecolor(COLOR_SURFACE)
ax_T.set_facecolor(COLOR_SURFACE)

ax_T.plot(detuning_MHz, probe_transmission, color=COLOR_GROUND, linewidth=2, zorder=3)

ax_T.set_xlabel("Détuning sonde δ / 2π (MHz)", fontsize=10, color=COLOR_MUTED)
ax_T.set_ylabel("Transmission", fontsize=10, color=COLOR_MUTED)
ax_T.set_title(
    f"Transmission de la sonde (L = {L_cell * 100:.2g} cm, N = {N_density:.1e} m⁻³)",
    fontsize=11, color=COLOR_INK, pad=10,
)
ax_T.set_ylim(0, 1.05)

ax_T.grid(axis="y", color=COLOR_GRID, linewidth=1, zorder=0)
ax_T.set_axisbelow(True)
for spine in ("top", "right"):
    ax_T.spines[spine].set_visible(False)
ax_T.spines["left"].set_color(COLOR_AXIS)
ax_T.spines["bottom"].set_color(COLOR_AXIS)
ax_T.tick_params(colors=COLOR_MUTED, labelsize=9)

fig3.tight_layout()

plt.show()