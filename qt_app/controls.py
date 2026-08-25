"""Parameter widgets: slider + spinbox pairs (linear or log) and a
scientific-notation text entry for values like the atomic density."""
import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QLineEdit, QSlider, QVBoxLayout, QWidget)


class FloatControl(QWidget):
    """A labeled slider synced with a spinbox. Set log=True for a log-scale slider."""

    valueChanged = Signal(float)

    def __init__(self, label, minimum, maximum, value, step=None, decimals=None,
                 log=False, parent=None):
        super().__init__(parent)
        self._min, self._max, self._log = float(minimum), float(maximum), log
        self._updating = False

        if log:
            self._lo, self._hi = math.log10(self._min), math.log10(self._max)
            self._n = 300
        else:
            step = step if step is not None else (self._max - self._min) / 100
            self._n = max(1, round((self._max - self._min) / step))
        if decimals is None:
            decimals = 3 if log else max(0, -math.floor(math.log10(step)))

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._n)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(self._min, self._max)
        self._spin.setDecimals(decimals)
        self._spin.setKeyboardTracking(False)
        if log:
            self._spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        else:
            self._spin.setSingleStep(step)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(self._spin)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 2, 0, 2)
        box.setSpacing(1)
        box.addWidget(QLabel(label))
        box.addLayout(row)

        self.setValue(value)
        self._slider.valueChanged.connect(self._from_slider)
        self._spin.valueChanged.connect(self._from_spin)

    def _slider_to_value(self, i):
        f = i / self._n
        if self._log:
            return 10 ** (self._lo + (self._hi - self._lo) * f)
        return self._min + (self._max - self._min) * f

    def _value_to_slider(self, v):
        if self._log:
            f = (math.log10(max(v, self._min)) - self._lo) / (self._hi - self._lo)
        else:
            f = (v - self._min) / (self._max - self._min)
        return round(min(max(f, 0.0), 1.0) * self._n)

    def _from_slider(self, i):
        if self._updating:
            return
        self._updating = True
        v = self._slider_to_value(i)
        self._spin.setValue(v)
        self._updating = False
        self.valueChanged.emit(self.value())

    def _from_spin(self, v):
        if self._updating:
            return
        self._updating = True
        self._slider.setValue(self._value_to_slider(v))
        self._updating = False
        self.valueChanged.emit(self.value())

    def value(self):
        return self._spin.value()

    def setValue(self, v):
        self._updating = True
        self._spin.setValue(v)
        self._slider.setValue(self._value_to_slider(v))
        self._updating = False


class ChoiceControl(QWidget):
    """A labeled combo box; value() returns the stored data of the current item."""

    valueChanged = Signal(str)

    def __init__(self, label, options, value, parent=None):
        super().__init__(parent)
        self._combo = QComboBox()
        for text, data in options:
            self._combo.addItem(text, data)
        self._combo.setCurrentIndex(max(0, self._combo.findData(value)))
        self._combo.currentIndexChanged.connect(
            lambda _: self.valueChanged.emit(self.value()))

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 2, 0, 2)
        box.setSpacing(1)
        box.addWidget(QLabel(label))
        box.addWidget(self._combo)

    def value(self):
        return self._combo.currentData()


class SciControl(QWidget):
    """A labeled line edit accepting scientific notation (e.g. 5.383e13)."""

    valueChanged = Signal(float)

    def __init__(self, label, value, parent=None):
        super().__init__(parent)
        self._value = float(value)
        self._edit = QLineEdit(f'{self._value:g}')
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.ScientificNotation)
        self._edit.setValidator(validator)
        self._edit.editingFinished.connect(self._commit)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 2, 0, 2)
        box.setSpacing(1)
        box.addWidget(QLabel(label))
        box.addWidget(self._edit)

    def _commit(self):
        try:
            v = float(self._edit.text().replace(',', '.'))
        except ValueError:
            self._edit.setText(f'{self._value:g}')
            return
        if v != self._value:
            self._value = v
            self.valueChanged.emit(v)

    def value(self):
        return self._value
