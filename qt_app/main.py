"""Main window: parameter panel on the left, plots on the right.

Every parameter change restarts a short debounce timer; the computation then
runs in a background thread (~0.35 s) and stale results are dropped, so the
sliders stay responsive.
"""
import sys
import time
import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QMainWindow,
                               QScrollArea, QSplitter, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from . import physics
from .controls import ChoiceControl, FloatControl, SciControl
from .plots import PopulationCanvas, RotationCanvas, SphereCanvas

# (group title, [(param name, control type, kwargs)])
CONTROL_SPEC = [
    ('Pump && cell', [
        ('pump_pol',      'choice', dict(label='Pump polarization',
                                         options=[('R  (σ⁻)', 'R'),
                                                  ('L  (σ⁺)', 'L'),
                                                  ('H', 'H'),
                                                  ('V', 'V'),
                                                  ('D', 'D'),
                                                  ('A', 'A')],
                                         value='R')),
        ('pump_s',        'log', dict(label='Pump saturation s = I/Isat',
                                      minimum=0.01, maximum=20.0, value=0.6)),
        ('pump_det_MHz',  'lin', dict(label='Pump detuning (MHz)',
                                      minimum=-30.0, maximum=30.0, step=0.5, value=0.0)),
        ('gamma_t',       'log', dict(label='Transit relaxation γ_t (rad/s)',
                                      minimum=1e3, maximum=1e6, value=3.0e4, decimals=0)),
        ('n_rho',         'sci', dict(label='Atomic density n (m^-3)', value=5.383e13)),
        ('L_cell_mm',     'sci', dict(label='Cell length L (mm)', value=75.0)),
    ]),
    ('B field', [
        ('B_norm_G',      'lin', dict(label='|B| (G)  — detuning scan',
                                      minimum=0.0, maximum=2.0, step=0.05, value=0.5)),
        ('theta_B_deg',   'lin', dict(label='B polar angle vs k (deg)',
                                      minimum=0.0, maximum=180.0, step=5.0, value=0.0)),
        ('phi_B_deg',     'lin', dict(label='B azimuth (deg)',
                                      minimum=0.0, maximum=360.0, step=5.0, value=0.0)),
    ]),
    ('Poincaré cut && scans', [
        ('cut_theta_deg', 'lin', dict(label='Cut normal: azimuth (deg)',
                                      minimum=0.0, maximum=360.0, step=5.0, value=0.0)),
        ('cut_phi_deg',   'lin', dict(label='Cut normal: polar (deg)',
                                      minimum=0.0, maximum=180.0, step=5.0, value=90.0)),
        ('det_scan_mode', 'choice', dict(label='Detuning applies to',
                                         options=[('Probe only (beat vs pump)', 'probe'),
                                                  ('Pump + probe together', 'both'),
                                                  ('Pump only (probe fixed)', 'pump')],
                                         value='probe')),
        ('det_max_MHz',   'sci', dict(label='Detuning scan range ± (MHz)', value=20.0)),
        ('B_max_G',       'sci', dict(label='B scan range ± (G)', value=1.0)),
        ('det_fixed_MHz', 'lin', dict(label='Fixed detuning of the B scan (MHz)',
                                      minimum=-30.0, maximum=30.0, step=0.5, value=0.0)),
        ('n_det',         'lin', dict(label='Detuning scan points',
                                      minimum=11, maximum=401, step=10, value=41)),
        ('n_B',           'lin', dict(label='B scan points',
                                      minimum=11, maximum=401, step=10, value=25)),
    ]),
]

DEBOUNCE_MS = 200


class _WorkerSignals(QObject):
    done = Signal(int, object, object, float)   # seq, params, results, elapsed
    failed = Signal(int, str)


class _ComputeJob(QRunnable):
    def __init__(self, seq, params, signals):
        super().__init__()
        self.seq, self.params, self.signals = seq, params, signals

    def run(self):
        try:
            t0 = time.time()
            results = physics.compute(self.params)
            self.signals.done.emit(self.seq, self.params, results, time.time() - t0)
        except RuntimeError:
            pass   # signals object already deleted: the window is closing
        except Exception:
            try:
                self.signals.failed.emit(self.seq, traceback.format_exc())
            except RuntimeError:
                pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rb85 Bloch — interactive simulator')

        self._controls = {}
        self._seq = 0
        self._pool = QThreadPool.globalInstance()
        self._signals = _WorkerSignals()
        self._signals.done.connect(self._on_done)
        self._signals.failed.connect(self._on_failed)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._launch)

        # left panel: parameter groups
        panel = QWidget()
        panel_box = QVBoxLayout(panel)
        for title, entries in CONTROL_SPEC:
            group = QGroupBox(title)
            group_box = QVBoxLayout(group)
            for name, kind, kwargs in entries:
                if kind == 'choice':
                    ctrl = ChoiceControl(**kwargs)
                elif kind == 'sci':
                    ctrl = SciControl(**kwargs)
                else:
                    ctrl = FloatControl(log=(kind == 'log'), **kwargs)
                ctrl.valueChanged.connect(self._schedule)
                group_box.addWidget(ctrl)
                self._controls[name] = ctrl
            panel_box.addWidget(group)
        panel_box.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)

        # right panel: plots (rotation plot gets most of the space)
        self.sphere_canvas = SphereCanvas()
        self.population_canvas = PopulationCanvas()
        self.rotation_canvas = RotationCanvas()

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.sphere_canvas)
        top.addWidget(self.population_canvas)

        right = QSplitter(Qt.Vertical)
        right.addWidget(top)
        right.addWidget(self.rotation_canvas)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 2)
        right.setSizes([330, 620])

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(scroll)
        layout.addWidget(right, stretch=1)
        self.setCentralWidget(central)

        self.statusBar().showMessage('Computing…')
        QTimer.singleShot(0, self._launch)

    def current_params(self):
        return physics.Params(**{name: ctrl.value()
                                 for name, ctrl in self._controls.items()})

    def _schedule(self, *_):
        self.statusBar().showMessage('Computing…')
        self._debounce.start()

    def _launch(self):
        self._seq += 1
        self._pool.start(_ComputeJob(self._seq, self.current_params(), self._signals))

    def _on_done(self, seq, params, results, elapsed):
        if seq != self._seq:
            return   # stale result, a newer computation is on its way
        self.apply_results(params, results)
        self.statusBar().showMessage(f'Up to date  ({elapsed:.2f} s)')

    def _on_failed(self, seq, message):
        if seq != self._seq:
            return
        print(message, file=sys.stderr)
        self.statusBar().showMessage('Computation failed — see terminal for the traceback')

    def closeEvent(self, event):
        self._debounce.stop()
        self._pool.waitForDone(3000)
        super().closeEvent(event)

    def apply_results(self, params, results):
        self.sphere_canvas.refresh(params, results)
        self.population_canvas.refresh(params, results)
        self.rotation_canvas.refresh(params, results)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1500, 950)
    window.show()
    sys.exit(app.exec())
