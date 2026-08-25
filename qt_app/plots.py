"""Matplotlib canvases embedded in Qt (QtAgg backend, no pyplot)."""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from . import physics
from .physics import AMP_THRESHOLD, COMPONENT_STYLES, wrap_pi


class BaseCanvas(FigureCanvas):
    def __init__(self, width, height):
        super().__init__(Figure(figsize=(width, height)))


class SphereCanvas(BaseCanvas):
    """Poincare sphere with the central cut + B direction vs the k axis."""

    def __init__(self):
        super().__init__(9, 4.2)

    def refresh(self, p, r):
        fig = self.figure
        fig.clf()

        ax = fig.add_subplot(121, projection='3d')
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        ax.plot_surface(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                        np.outer(np.ones_like(u), np.cos(v)),
                        color='lightgray', alpha=0.12, linewidth=0)
        ax.plot(np.cos(u), np.sin(u), np.zeros_like(u), color='gray', linewidth=1)
        ax.plot(np.cos(u), np.zeros_like(u), np.sin(u), color='gray', linewidth=1)
        ax.plot(np.zeros_like(u), np.cos(u), np.sin(u), color='gray', linewidth=1)
        t = np.linspace(0, 2 * np.pi, 120)
        circle = np.outer(np.cos(t), r.e1) + np.outer(np.sin(t), r.e2)
        ax.plot(circle[:, 0], circle[:, 1], circle[:, 2],
                color='darkred', linewidth=2, label='Central cut')
        for label, (x0, y0, z0) in dict(H=(1, 0, 0), V=(-1, 0, 0), D=(0, 1, 0),
                                        A=(0, -1, 0), R=(0, 0, 1), L=(0, 0, -1)).items():
            ax.scatter([x0], [y0], [z0], color='white', s=100,
                       edgecolors='black', linewidths=0.8)
            ax.text(x0 * 1.15, y0 * 1.15, z0 * 1.15, label,
                    color='black', fontsize=11, ha='center', va='center')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Poincaré sphere')
        ax.legend(loc='upper left', fontsize=8)

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.quiver(0, 0, 0, 0, 0, 1, color='purple', label='Propagation (pump + probe)')
        Bv = r.B_dir * max(p.B_norm_G, 1e-9)
        ax2.quiver(0, 0, 0, Bv[0], Bv[1], Bv[2], color='blue', label=f'B ({p.B_norm_G:.2f} G)')
        lim = max(1.0, p.B_norm_G)
        ax2.set_xlim([-lim, lim]); ax2.set_ylim([-lim, lim]); ax2.set_zlim([-lim, lim])
        ax2.set_box_aspect([1, 1, 1])
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        ax2.set_title('B field in the (e1, e2, k) frame')
        ax2.legend(loc='upper left', fontsize=8)

        fig.tight_layout()
        self.draw_idle()


class PopulationCanvas(BaseCanvas):
    """Steady-state population distribution (F=3 / F'=4)."""

    def __init__(self):
        super().__init__(8, 4.2)

    def refresh(self, p, r):
        fig = self.figure
        fig.clf()
        ground = r.populations[:physics.N_G]
        excited = r.populations[physics.N_G:]
        axg, axe = fig.subplots(1, 2, sharey=True)
        for ax, m_values, pop, title, color in (
            (axg, physics.MF3, ground, 'Ground state (F = 3)', '#2a78d6'),
            (axe, physics.MFP4, excited, "Excited state (F' = 4)", '#eb6834'),
        ):
            ax.bar(m_values, pop, width=0.6, color=color, zorder=3)
            top = int(np.argmax(pop))
            ax.annotate(f'{pop[top]:.3f}', (m_values[top], pop[top]),
                        textcoords='offset points', xytext=(0, 4),
                        ha='center', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(r'$m_F$')
            ax.set_xticks(m_values)
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
        axg.set_ylabel('Population')
        fig.suptitle(f'Steady-state populations (pump {p.pump_pol})  —  '
                     f"F=3: {ground.sum():.3f},  F'=4: {excited.sum():.3f}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        self.draw_idle()


class RotationCanvas(BaseCanvas):
    """The two rotation plots: fringe phase/amplitude vs detuning and vs signed B."""

    def __init__(self):
        super().__init__(14, 8)

    def refresh(self, p, r):
        fig = self.figure
        fig.clf()
        axs = fig.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(f'Rotation and amplitude  —  '
                     f'pump {p.pump_pol}, s = {p.pump_s:.2f},  '
                     f'left: B = {p.B_norm_G:.2f} G  |  '
                     f'right: detuning = {p.det_fixed_MHz:g} MHz', fontsize=12)

        det_xlabel = {'probe': 'Probe detuning (MHz)',
                      'both': 'Pump + probe detuning (MHz)',
                      'pump': 'Pump detuning (MHz)'}[p.det_scan_mode]
        for col, (x, phases, amps, xlabel) in enumerate([
            (r.det_MHz, r.ph_det, r.am_det, det_xlabel),
            (r.B_scan_G, r.ph_B, r.am_B, 'Signed B (G)'),
        ]):
            ax_ph, ax_am = axs[0, col], axs[1, col]
            for key, label, color, ls in COMPONENT_STYLES:
                kw = dict(marker='o', markersize=3, linewidth=1.8,
                          color=color, linestyle=ls, label=label)
                if np.nanmean(amps[key]) > AMP_THRESHOLD:
                    ax_ph.plot(x, np.degrees(wrap_pi(phases[key])), **kw)
                ax_am.plot(x, amps[key], **kw)
            for ax in (ax_ph, ax_am):
                ax.axhline(0, color='black', linewidth=0.8, alpha=0.45)
                ax.axvline(0, color='black', linewidth=0.8, alpha=0.45)
                ax.grid(alpha=0.25)
                ax.legend(frameon=False, ncol=3, fontsize=8)
            ax_ph.set_ylabel(r'$\phi - \phi(0)$ mod $2\pi$ (deg)')
            ax_am.set_ylabel('Amplitude $A$')
            ax_am.set_xlabel(xlabel)
            ax_am.set_xlim(x.min(), x.max())

        axs[0, 0].set_title('Rotation vs detuning (fixed B)', fontsize=11)
        axs[0, 1].set_title('Rotation vs signed B (fixed detuning)', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        self.draw_idle()
