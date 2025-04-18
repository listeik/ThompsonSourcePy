import sys
import os
import json
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFrame, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QComboBox, QPushButton, QLineEdit, QLabel, QCheckBox, QProgressBar,
                             QTabWidget, QScrollArea, QSlider, QMenuBar, QMenu, QAction,
                             QRadioButton, QButtonGroup, QFileDialog, QMessageBox, QFormLayout,
                             QWidget, QSizePolicy, QTextEdit, QTextBrowser, QDialog, QScrollArea,QActionGroup)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QDoubleValidator, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from decimal import Decimal, getcontext

# Import classes from other files
from part1 import ColorChartParam
from part2 import ElectronBunch
from part3 import JetPaintScale
from part4 import LaserPulse
from part5 import LinearChartParam
from part6 import ThompsonSource, Vector

from typing import List, Optional

from PyQt5.QtGui import QIntValidator
import logging
from typing import Dict, Union, Optional


class MyTextUtilities:
    @staticmethod
    def test_value(min_val: float, max_val: float, field: QLineEdit, default_str: str) -> float:
        text = field.text()
        if not text:
            return 0.0

        try:
            value = float(text)
        except ValueError:
            field.setText(default_str)
            return float(default_str)

        if not (min_val <= value <= max_val):
            field.setText(default_str)
            return float(default_str)
        return value

    @staticmethod
    def get_integer_field(value: int, min_val: int, max_val: int) -> QLineEdit:
        edit = QLineEdit(str(value))
        edit.setValidator(QIntValidator(min_val, max_val))
        edit.setAlignment(Qt.AlignRight)
        return edit

    @staticmethod
    def get_double_field(value: float, min_val: float, max_val: float, scientific=False) -> QLineEdit:
        edit = QLineEdit()
        if scientific:
            edit.setText("{:.4e}".format(value))
        else:
            edit.setText("{:.4f}".format(value))

        validator = QDoubleValidator(min_val, max_val, 6)
        validator.setNotation(QDoubleValidator.ScientificNotation if scientific
                              else QDoubleValidator.StandardNotation)
        edit.setValidator(validator)
        edit.setAlignment(Qt.AlignRight)
        return edit

class CalcBoxParam(QObject):
    MIN_DIF = 1.0E-10

    def __init__(self, keys: List[str], parent=None):
        super().__init__(parent)
        self.keys = keys
        self.valueUnitLabels: List[str] = []
        self.plotLabels: List[str] = []
        self.minValues: List[str] = []
        self.maxValues: List[str] = []
        self.selectedItemIndex = 10
        self.selectedItemIndexClone = 10
        self.numberOfItems = 0
        self.minValue = 0.0
        self.maxValue = 100.0
        self.conversionValues: List[float] = []
        self.tsourceclone = None
        self.espread = False
        self.working = False
        self.savetext = ""
        self.chartParam = LinearChartParam()
        self.chartPanel = None
        self.chart = None
        self.angle = 0.0
        self.energy = 46.0
        self.file = None
        self.parent = parent

    def initialize(self):
        self.working = True
        try:
            self.tsourceclone = self.parent.tsource.clone()
        except:
            self.tsourceclone = ThompsonSource(self.parent.lpulse, self.parent.ebunch)

        self.tsourceclone.seteSpread(self.espread)
        self.minValueClone = self.minValue
        self.maxValueClone = self.maxValue
        self.angleclone = self.angle
        self.energyclone = self.energy
        self.selectedItemIndexClone = self.selectedItemIndex

    def save(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            None,
            self.savetext,
            "",
            "All Files (*);;Text Files (*.txt)",
            options=options
        )

        if not file_name:
            return

        if os.path.exists(file_name):
            reply = QMessageBox.question(
                None,
                'Warning',
                "The file already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        try:
            with open(file_name, 'w') as f:
                nc = len(self.chartParam.getData())
                for i in range(self.chartParam.getSize()):
                    line = f"{i * self.chartParam.getStep() + self.chartParam.getOffset():10.3f}"
                    for data in self.chartParam.getData():
                        line += f" {data[i]:10.3f}"
                    f.write(line + "\n")
        except IOError as e:
            QMessageBox.critical(None, "Error", "Error while writing to the file")


class ConcreteColorChartParam(ColorChartParam):
    def __init__(self, func):
        super().__init__()
        self._func = func

    def func(self, x: float, y: float) -> float:
        return self._func(x, y)


class ThomsonJFrame(QMainWindow):
    def __init__(self):
        super().__init__()

        # Инициализация переменных
        self.tsourceRayClone = None
        self.sliderposition = 50
        self.hoffset = 0.0
        self.numberOfRays = 1000
        self.xenergycrosschart = None
        self.working = False
        self.rayWorking = False
        self.bFile = None
        self.pFile = None

        self.ebunch = ElectronBunch()
        self.lpulse = LaserPulse()
        self.tsource = ThompsonSource(self.lpulse, self.ebunch)
        self.tsource.set_polarization([0.0, 0.0, 0.0])

        self.xsize = 300
        self.ysize = 200
        self.xstep = 20.0 / self.xsize
        self.ystep = 20.0 / self.ysize
        self.estep = 2000 / self.xsize
        self.oldStrings = {}

        # Создание полей ввода с помощью MyTextUtilities
        self.rayNumberBox = MyTextUtilities.get_integer_field(1000, 1, 1000000)
        self.rayXAngleRangeBox = MyTextUtilities.get_double_field(0.5, 0.0, 100.0)
        self.rayYAngleRangeBox = MyTextUtilities.get_double_field(0.5, 0.0, 100.0)
        self.numericallPrecisionBox = MyTextUtilities.get_double_field(1.0e-4, 1.0e-10, 0.1, True)

        self.orderofmagnitude = 10
        self.normfactor = 10 ** (-15 - self.orderofmagnitude)

        # Инициализация форм
        self._init_forms()

        # Инициализация UI
        self.initUI()
        self.initChartParams()
        self.loadParameters()

    def _init_forms(self):
        self.brilForm = CalcBoxParam(["Spectral brilliance"], self)
        self.brilForm.valueUnitLabels = ["mrad", "ps", "mm", "mm", "mm mrad", "mm mrad", "mm mrad",
                                         "mm", "μm", "", "keV", "mrad"]
        self.brilForm.plotLabels = ["Laser-electron angle, mrad", "Delay, ps", "Z-shift, mm",
                                    "beta, mm", "eps, mm mrad", "X-eps, mm mrad", "Y-eps, mm mrad",
                                    "Reyleigh length, mm", "Waist semi-width, μm", "Δγ/γ",
                                    "X-ray energy, keV", "Observation angle, mrad"]
        self.brilForm.conversionValues = [0.001, 3.0E-4, 0.001, 0.001, 1.0E-6, 1.0E-6, 1.0E-6,
                                          0.001, 1.0E-6, 0.01, 1.602E-16, 0.001]
        self.brilForm.minValues = ["0", "0", "0", "10", "0.5", "0.5", "0.5", "0.3", "5", "0.1", "0", "0"]
        self.brilForm.maxValues = ["50", "100", "10", "50", "5", "5", "5", "3", "50", "1", "100", "5"]
        self.brilForm.savetext = "Choose file to save spectral brilliance data"
        self.brilForm.numberOfItems = 12

        self.gfForm = CalcBoxParam(["Full flux", "Approximate full flux"], self)
        self.gfForm.valueUnitLabels = ["mrad", "ps", "mm", "mm", "mm mrad", "mm", "μm"]
        self.gfForm.plotLabels = ["Angle, mrad", "Delay, ps", "Z-shift, mm", "beta, mm",
                                  "eps, mm mrad", "Reyleigh length, mm", "Waist semi-width, μm"]
        self.gfForm.conversionValues = [0.001, 3.0E-4, 0.001, 0.001, 1.0E-6, 0.001, 1.0E-6]
        self.gfForm.minValues = ["0", "0", "0", "10", "0.5", "0.3", "5"]
        self.gfForm.maxValues = ["50", "100", "10", "50", "5", "3", "50"]
        self.gfForm.savetext = "Choose file to save geometric factor data"
        self.gfForm.numberOfItems = 7
        self.gfForm.selectedItemIndex = 0
        self.gfForm.selectedItemIndexClone = 0

        self.polForm = CalcBoxParam(["ξ1", "ξ2", "ξ3", "polarization degree"], self)
        self.polForm.valueUnitLabels = ["mrad", "ps", "mm", "mm", "mm mrad", "mm mrad", "mm mrad",
                                        "mm", "μm", "", "keV", "mrad"]
        self.polForm.plotLabels = ["Laser-electron angle, mrad", "Delay, ps", "Z-shift, mm",
                                   "beta, mm", "eps, mm mrad", "X-eps, mm mrad", "Y-eps, mm mrad",
                                   "Reyleigh length, mm", "Waist semi-width, μm", "Δγ/γ",
                                   "X-ray energy, keV", "Observation angle, mrad"]
        self.polForm.conversionValues = [0.001, 3.0E-4, 0.001, 0.001, 1.0E-6, 1.0E-6, 1.0E-6,
                                         0.001, 1.0E-6, 0.01, 1.602E-16, 0.001]
        self.polForm.minValues = ["0", "0", "0", "10", "0.5", "0.5", "0.5", "0.3", "5", "0.1", "0", "0"]
        self.polForm.maxValues = ["50", "100", "10", "50", "5", "5", "5", "3", "50", "1", "100", "5"]
        self.polForm.savetext = "Choose file to save polarization data"
        self.polForm.numberOfItems = 12

        self.paramNames = [
            "Electron_energy_MeV", "Electron_bunch_charge_nQ",
            "Electron_bunch_relative_energy_spread_%", "Electron_bunch_length_ps",
            "X-emittance_mm*mrad", "Y-emittance_mm*mrad", "Beta-x_function_mm",
            "Beta-y_function_mm", "Photon_energy_eV", "Pulse_energy_mJ",
            "Laser_pulse_length_ps", "Rayleigh_length_mm", "Pulse_frequency_MHz",
            "Delay_ps", "X-shift_mm", "Y-shift_mm", "Z-shift_mm",
            "Laser-electron_angle_mrad"
        ]

    def initChartParams(self):
        self.fluxdata = ConcreteColorChartParam(
            lambda thetax, thetay: 9.999999999999999E-14 * self.tsource.direction_flux(
                Vector(thetax * 0.001, thetay * 0.001, 1.0).normalize(),
                Vector(0.0, 0.0, 1.0)
            )
        )

        self.fluxcrossdata = ConcreteColorChartParam(
            lambda e, theta: 1.0E-16 * self.tsource.geometric_factor *
                             self.tsource.direction_frequency_flux_no_spread(
                                 Vector(self.hoffset * 0.001, theta * 0.001, 1.0).normalize(),
                                 Vector(0.0, 0.0, 1.0),
                                 e * 1.602E-19
                             )
        )

        self.xenergydata = ConcreteColorChartParam(
            lambda thetax, thetay: self.tsource.direction_energy(
                Vector(thetax * 0.001, thetay * 0.001, 1.0).normalize(),
                Vector(0.0, 0.0, 1.0)
            ) / 1.602E-19 * 0.001
        )

        self.xenergycrossdata = LinearChartParam()
        # # Create matplotlib figure and canvas
        # self.figure = Figure()
        # self.canvas = FigureCanvas(self.figure)
        #
        # # Add canvas to the main panel
        # self.jPanel_xflux_left.layout().addWidget(self.canvas)

    def BrillianceCalcStartActionPerformed(self, evt=None):
        self.brilForm.initialize()
        self.plotBrillianceChart()

    # Реализация отрисовки графика блеска
    def plotBrillianceChart(self):
        self.brilliance_figure.clear()
        ax = self.brilliance_figure.add_subplot(111)

        # Получаем параметры от пользователя
        try:
            min_val = float(self.Brilminvalue.text())
            max_val = float(self.Brilmaxvalue.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Min/Max values must be numbers.")
            return

        # Генерация данных для оси X
        x = np.linspace(min_val, max_val, 300)

        # Установка параметров источника
        self.tsource.seteSpread(self.jCheckBoxSpread.isChecked())
        self.tsource.setEnergy(float(self.energyValue.text()) * 1.602E-16)
        angle_rad = float(self.angleValue.text()) * 1e-3
        direction = Vector(0.0, np.sin(angle_rad), np.cos(angle_rad))

        # Выбор функции (по текущему индексу выпадающего списка)
        index = self.BrillianceCalcBox.currentIndex()

        # Просто пример функции — здесь ты должен указать корректную функцию расчёта значения
        y = []
        for xi in x:
            try:
                # примерная заглушка — здесь должен быть вызов нужной функции блеска
                val = self.tsource.spectral_brilliance(direction, xi * 1e-3, float(self.energyValue.text()) * 1.602e-16)
            except Exception as e:
                val = 0
            y.append(val * self.normfactor)

        ax.plot(x, y)
        ax.set_xlabel(self.brilForm.plotLabels[index])
        ax.set_ylabel("Spectral Brilliance (arb. units)")
        ax.set_title("Spectral Brilliance vs " + self.brilForm.plotLabels[index])

        self.brilliance_canvas.draw()

    def BrillianceCalcBoxActionPerformed(self, index=None):
        if index is None:
            index = self.BrillianceCalcBox.currentIndex()
        self.brilForm.selectedItemIndex = index

    def BrilminvalueFocusLost(self):
        val = MyTextUtilities.test_value(0, 1000, self.Brilminvalue, "0")
        self.brilForm.minValue = val

    def BrilmaxvalueFocusLost(self):
        val = MyTextUtilities.test_value(0, 1000, self.Brilmaxvalue, "100")
        self.brilForm.maxValue = val

    def angleValueFocusLost(self):
        val = MyTextUtilities.test_value(0, 1000, self.angleValue, "0")
        self.brilForm.angle = val

    def energyValueFocusLost(self):
        val = MyTextUtilities.test_value(0, 1000, self.energyValue, "46")
        self.brilForm.energy = val

    def GFCalcStartActionPerformed(self, evt=None):
        self.gfForm.initialize()
        self.plotGFChart()

    def plotGFChart(self):
        self.gf_figure.clear()
        ax = self.gf_figure.add_subplot(111)

        try:
            min_val = float(self.GFminvalue.text())
            max_val = float(self.GFmaxvalue.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Min/Max values must be numbers.")
            return

        x = np.linspace(min_val, max_val, 300)
        self.tsource.seteSpread(False)
        index = self.GFCalcBox.currentIndex()
        mode = self.GFValueSelectionBox.currentIndex()

        y = []
        for xi in x:
            try:
                val = (self.tsource.full_flux(xi * 1e-3) if mode == 0
                       else self.tsource.geometric_factor_at(xi * 1e-3))
            except Exception:
                val = 0
            y.append(val * self.normfactor)

        ax.plot(x, y)
        ax.set_xlabel(self.gfForm.plotLabels[index])
        ax.set_ylabel("Value")
        ax.set_title("Full Flux / Geometric Factor")
        self.gf_canvas.draw()

    # === Polarization ===
    def polarizationCalcStartActionPerformed(self, evt=None):
        self.polForm.initialize()
        self.plotPolarizationChart()

    def plotPolarizationChart(self):
        self.pol_figure.clear()
        ax = self.pol_figure.add_subplot(111)

        try:
            min_val = float(self.polminvalue.text())
            max_val = float(self.polmaxvalue.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Min/Max values must be numbers.")
            return

        x = np.linspace(min_val, max_val, 300)
        self.tsource.seteSpread(self.jPolCheckBoxSpread.isChecked())
        self.tsource.setEnergy(float(self.polEnergyValue.text()) * 1.602E-16)
        angle_rad = float(self.polAngleValue.text()) * 1e-3
        direction = Vector(0.0, np.sin(angle_rad), np.cos(angle_rad))
        index = self.polarizationCalcBox.currentIndex()

        y = []
        for xi in x:
            try:
                val = self.tsource.polarization_component(index, direction, xi * 1e-3)
            except Exception:
                val = 0
            y.append(val)

        ax.plot(x, y)
        ax.set_xlabel(self.polForm.plotLabels[index])
        ax.set_ylabel("Polarization Component")
        ax.set_title("Polarization vs " + self.polForm.plotLabels[index])
        self.pol_canvas.draw()

    def initUI(self):
        self.setWindowTitle("TSourceXG")
        self.setMinimumSize(750, 600)

        # Create main widgets
        self.createBrillianceCalcFrame()
        self.brilliance_figure = Figure()
        self.brilliance_canvas = FigureCanvas(self.brilliance_figure)

        layout = QVBoxLayout()
        layout.addWidget(self.brilliance_canvas)
        self.BrillianceCalcGraph.setLayout(layout)
        self.createGFCalcFrame()
        self.createPolarizationCalcFrame()
        self.createRayProgressFrame()
        self.createMainContent()
        self.createMenuBar()

        # Set central widget
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)  # Сначала установите layout

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(central_widget)  # Затем добавьте в QScrollArea

        self.setCentralWidget(scroll_area)




    def loadParameters(self):
        config_path = os.path.join(os.getcwd(), "my.ini")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    params = json.load(f)
                    # Load parameters from file
                    # Implementation depends on your parameter file structure
                    pass
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load parameters: {str(e)}")

    def createBrillianceCalcFrame(self):
        self.brillianceCalc = QFrame()
        self.brillianceCalc.setWindowTitle("Brilliance box")
        self.brillianceCalc.setMinimumSize(800, 500)

        # Brilliance Param Panel
        self.BrillianceParam = QGroupBox("Plot parameter selection")

        self.BrillianceCalcBox = QComboBox()
        self.BrillianceCalcBox.addItems([
            "Laser-electron angle", "Delay", "Z-shift", "Beta function", "Emittance",
            "X-emittance", "Y-emittance", "Rayleigh length", "Waist semi-width", "Energy spread",
            "X-ray energy", "Observation angle"
        ])
        self.BrillianceCalcBox.setCurrentIndex(10)
        self.BrillianceCalcBox.currentIndexChanged.connect(self.BrillianceCalcBoxActionPerformed)

        self.BrillianceCalcStart = QPushButton("Calculate")
        self.BrillianceCalcStart.clicked.connect(self.BrillianceCalcStartActionPerformed)

        self.BrillianceCalcSave = QPushButton("Save")
        self.BrillianceCalcSave.clicked.connect(self.BrillianceCalcSaveActionPerformed)

        self.Brilminvalue = QLineEdit("0")
        self.Brilminvalue.setValidator(QDoubleValidator())
        self.Brilminvalue.editingFinished.connect(self.BrilminvalueFocusLost)

        self.Brilminvaluelabel = QLabel("Min value")
        self.Brilminvalueunitlabel = QLabel("keV")

        self.Brilmaxvalue = QLineEdit("100")
        self.Brilmaxvalue.setValidator(QDoubleValidator())
        self.Brilmaxvalue.editingFinished.connect(self.BrilmaxvalueFocusLost)

        self.Brilmaxvaluelabel = QLabel("Max value")
        self.Brilmaxvalueunitlabel = QLabel("keV")

        self.jCheckBoxSpread = QCheckBox("Spread")
        self.jCheckBoxSpread.stateChanged.connect(self.jCheckBoxSpreadActionPerformed)

        self.jAngleLabel = QLabel("Angle")
        self.angleValue = QLineEdit("0")
        self.angleValue.setValidator(QDoubleValidator())
        self.angleValue.editingFinished.connect(self.angleValueFocusLost)
        self.angleValueUnitLable = QLabel("mrad")

        self.jEnergyLabel = QLabel("Energy")
        self.energyValue = QLineEdit("46")
        self.energyValue.setValidator(QDoubleValidator())
        self.energyValue.setEnabled(False)
        self.energyValue.editingFinished.connect(self.energyValueFocusLost)
        self.energyValueUnitLable = QLabel("keV")

        self.BrilProgressBar = QProgressBar()

        # Layout for BrillianceParam
        param_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        top_row.addWidget(self.BrillianceCalcBox)
        top_row.addWidget(self.BrillianceCalcStart)
        top_row.addWidget(self.BrillianceCalcSave)
        top_row.addWidget(self.jAngleLabel)
        top_row.addWidget(self.angleValue)
        top_row.addWidget(self.angleValueUnitLable)
        param_layout.addLayout(top_row)

        mid_row = QHBoxLayout()
        mid_row.addWidget(self.Brilminvaluelabel)
        mid_row.addWidget(self.Brilminvalue)
        mid_row.addWidget(self.Brilminvalueunitlabel)
        mid_row.addWidget(self.Brilmaxvaluelabel)
        mid_row.addWidget(self.Brilmaxvalue)
        mid_row.addWidget(self.Brilmaxvalueunitlabel)
        mid_row.addWidget(self.jCheckBoxSpread)
        param_layout.addLayout(mid_row)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.BrilProgressBar)
        bottom_row.addWidget(self.jEnergyLabel)
        bottom_row.addWidget(self.energyValue)
        bottom_row.addWidget(self.energyValueUnitLable)
        param_layout.addLayout(bottom_row)

        self.BrillianceParam.setLayout(param_layout)

        # BrillianceCalcGraph Panel
        self.BrillianceCalcGraph = QGroupBox("Spectral brilliance")
        self.BrillianceCalcGraph.setMinimumSize(639, 215)

        # Main layout for brillianceCalc
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.BrillianceParam)
        main_layout.addWidget(self.BrillianceCalcGraph)

        self.brillianceCalc.setLayout(main_layout)

    def createGFCalcFrame(self):
        self.gfCalc = QFrame()
        self.gfCalc.setWindowTitle("Full flux box")
        self.gfCalc.setMinimumSize(800, 500)

        # GF Param Panel
        self.GFParam = QGroupBox("Plot parameter selection")

        self.GFCalcBox = QComboBox()
        self.GFCalcBox.addItems([
            "Laser-electron angle", "Delay", "Z-shift", "Beta function",
            "Emittance", "Rayleigh length", "Waist semi-width"
        ])
        self.GFCalcBox.currentIndexChanged.connect(self.GFCalcBoxActionPerformed)

        self.GFCalcStart = QPushButton("Calculate")
        self.GFCalcStart.clicked.connect(self.GFCalcStartActionPerformed)

        self.GFCalcSave = QPushButton("Save")
        self.GFCalcSave.clicked.connect(self.GFCalcSaveActionPerformed)

        self.GFminvalue = QLineEdit("0")
        self.GFminvalue.setValidator(QDoubleValidator())
        self.GFminvalue.editingFinished.connect(self.GFminvalueFocusLost)

        self.GFminvaluelabel = QLabel("Min value")
        self.GFminvalueunitlabel = QLabel("mrad")

        self.GFmaxvalue = QLineEdit("60")
        self.GFmaxvalue.setValidator(QDoubleValidator())
        self.GFmaxvalue.editingFinished.connect(self.GFmaxvalueFocusLost)

        self.GFmaxvaluelabel = QLabel("Max value")
        self.GFmaxvalueunitlabel = QLabel("mrad")

        self.GFValueSelectionBox = QComboBox()
        self.GFValueSelectionBox.addItems(["Full flux", "Geometric factor"])
        self.GFValueSelectionBox.currentIndexChanged.connect(self.GFValueSelectionBoxActionPerformed)

        self.GFProgressBar = QProgressBar()

        # Layout for GFParam
        param_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        top_row.addWidget(self.GFCalcBox)
        top_row.addWidget(self.GFCalcStart)
        top_row.addWidget(self.GFCalcSave)
        param_layout.addLayout(top_row)

        mid_row = QHBoxLayout()
        mid_row.addWidget(self.GFminvaluelabel)
        mid_row.addWidget(self.GFminvalue)
        mid_row.addWidget(self.GFminvalueunitlabel)
        mid_row.addWidget(self.GFmaxvaluelabel)
        mid_row.addWidget(self.GFmaxvalue)
        mid_row.addWidget(self.GFmaxvalueunitlabel)
        mid_row.addWidget(self.GFValueSelectionBox)
        param_layout.addLayout(mid_row)

        param_layout.addWidget(self.GFProgressBar)

        self.GFParam.setLayout(param_layout)

        # GFCalcGraph Panel
        self.GFCalcGraph = QGroupBox("Full flux")
        self.GFCalcGraph.setMinimumSize(418, 216)

        # Main layout for gfCalc
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.GFParam)
        main_layout.addWidget(self.GFCalcGraph)

        # Create figure and canvas for GF plot
        self.gf_figure = Figure()
        self.gf_canvas = FigureCanvas(self.gf_figure)

        gf_layout = QVBoxLayout()
        gf_layout.addWidget(self.gf_canvas)
        self.GFCalcGraph.setLayout(gf_layout)

        self.gfCalc.setLayout(main_layout)

    def createPolarizationCalcFrame(self):
        self.polarizationCalc = QFrame()
        self.polarizationCalc.setWindowTitle("Polarization box")
        self.polarizationCalc.setMinimumSize(800, 500)

        # Polarization Param Panel
        self.polarizationParam = QGroupBox("Plot parameter selection")

        self.polarizationCalcBox = QComboBox()
        self.polarizationCalcBox.addItems([
            "Laser-electron angle", "Delay", "Z-shift", "Beta function", "Emittance",
            "X-emittance", "Y-emittance", "Rayleigh length", "Waist semi-width", "Energy spread",
            "X-ray energy", "Observation angle"
        ])
        self.polarizationCalcBox.setCurrentIndex(10)
        self.polarizationCalcBox.currentIndexChanged.connect(self.polarizationCalcBoxActionPerformed)

        self.polarizationCalcStart = QPushButton("Calculate")
        self.polarizationCalcStart.clicked.connect(self.polarizationCalcStartActionPerformed)

        self.polarizationCalcSave = QPushButton("Save")
        self.polarizationCalcSave.clicked.connect(self.polarizationCalcSaveActionPerformed)

        self.polminvalue = QLineEdit("0")
        self.polminvalue.setValidator(QDoubleValidator())
        self.polminvalue.editingFinished.connect(self.polminvalueFocusLost)

        self.polminvaluelabel = QLabel("Min value")
        self.polminvalueunitlabel = QLabel("keV")

        self.polmaxvalue = QLineEdit("100")
        self.polmaxvalue.setValidator(QDoubleValidator())
        self.polmaxvalue.editingFinished.connect(self.polmaxvalueFocusLost)

        self.polmaxvaluelabel = QLabel("Max value")
        self.polmaxvalueunitlabel = QLabel("keV")

        self.jPolCheckBoxSpread = QCheckBox("Spread")
        self.jPolCheckBoxSpread.stateChanged.connect(self.jPolCheckBoxSpreadActionPerformed)

        self.jPolAngleLabel = QLabel("Angle")
        self.polAngleValue = QLineEdit("0")
        self.polAngleValue.setValidator(QDoubleValidator())
        self.polAngleValue.editingFinished.connect(self.polAngleValueFocusLost)
        self.polAngleValueUnitLable = QLabel("mrad")

        self.jPolEnergyLabel = QLabel("Energy")
        self.polEnergyValue = QLineEdit("46")
        self.polEnergyValue.setValidator(QDoubleValidator())
        self.polEnergyValue.setEnabled(False)
        self.polEnergyValue.editingFinished.connect(self.polEnergyValueFocusLost)
        self.polEnergyValueUnitLable = QLabel("keV")

        self.polProgressBar = QProgressBar()

        # Layout for polarizationParam
        param_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        top_row.addWidget(self.polarizationCalcBox)
        top_row.addWidget(self.polarizationCalcStart)
        top_row.addWidget(self.polarizationCalcSave)
        top_row.addWidget(self.jPolAngleLabel)
        top_row.addWidget(self.polAngleValue)
        top_row.addWidget(self.polAngleValueUnitLable)
        param_layout.addLayout(top_row)

        mid_row = QHBoxLayout()
        mid_row.addWidget(self.polminvaluelabel)
        mid_row.addWidget(self.polminvalue)
        mid_row.addWidget(self.polminvalueunitlabel)
        mid_row.addWidget(self.polmaxvaluelabel)
        mid_row.addWidget(self.polmaxvalue)
        mid_row.addWidget(self.polmaxvalueunitlabel)
        mid_row.addWidget(self.jPolCheckBoxSpread)
        param_layout.addLayout(mid_row)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.polProgressBar)
        bottom_row.addWidget(self.jPolEnergyLabel)
        bottom_row.addWidget(self.polEnergyValue)
        bottom_row.addWidget(self.polEnergyValueUnitLable)
        param_layout.addLayout(bottom_row)

        self.polarizationParam.setLayout(param_layout)

        # PolarizationCalcGraph Panel
        self.polarizationCalcGraph = QGroupBox("Polarization parameters")
        self.polarizationCalcGraph.setMinimumSize(639, 215)

        # Main layout for polarizationCalc
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.polarizationParam)
        main_layout.addWidget(self.polarizationCalcGraph)

        self.polarizationCalc.setLayout(main_layout)

    def createRayProgressFrame(self):
        self.rayProgressFrame = QFrame()
        self.rayProgressFrame.setWindowTitle("Ray generation progress")
        self.rayProgressFrame.setWindowFlags(self.rayProgressFrame.windowFlags() | Qt.WindowStaysOnTopHint)
        self.rayProgressFrame.setMinimumSize(400, 100)
        self.rayProgressFrame.setFixedSize(400, 100)

        self.jRayProgressBar = QProgressBar()
        self.jRayStopButton = QPushButton("Stop")
        self.jRayStopButton.setEnabled(False)
        self.jRayStopButton.clicked.connect(self.jRayStopButtonActionPerformed)

        self.jLabelPartialFlux = QLabel("Flux: ")

        layout = QVBoxLayout()
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.jRayProgressBar)
        progress_row.addWidget(self.jRayStopButton)

        layout.addLayout(progress_row)
        layout.addWidget(self.jLabelPartialFlux)

        self.rayProgressFrame.setLayout(layout)

    def createMainContent(self):
        # Main panel
        self.jPanel1 = QWidget()
        self.jPanel1.setMinimumSize(720, 0)

        # Create electron bunch panel
        self.createElectronBunchPanel()

        # Create laser pulse panel
        self.createLaserPulsePanel()

        # Create relative position panel
        self.createRelativePositionPanel()

        # Create execution panel
        self.createExecutionPanel()

        # Create tabbed pane
        self.createTabbedPane()

        # Create slider panel
        self.createSliderPanel()

        # Main layout
        main_layout = QVBoxLayout()

        # Top row with parameter panels
        top_row = QHBoxLayout()
        top_row.addWidget(self.jPanel_el)
        top_row.addWidget(self.jPanel_ph)

        right_column = QVBoxLayout()
        right_column.addWidget(self.jPanel_sh)
        right_column.addWidget(self.jPanel_exec)

        top_row.addLayout(right_column)
        main_layout.addLayout(top_row)

        # Add tabbed pane
        main_layout.addWidget(self.jTabbedPane1)

        # Add slider panel
        main_layout.addWidget(self.jPanel_slider)

        self.jPanel1.setLayout(main_layout)

    def createElectronBunchPanel(self):
        self.jPanel_el = QGroupBox("Electron bunch parameters")
        self.jPanel_el.setMinimumSize(247, 350)

        # Create widgets
        self.energylabel = QLabel("Electron energy")
        self.energyvalue = QLineEdit("50")
        self.energyvalue.setValidator(QDoubleValidator())
        self.energyvalue.editingFinished.connect(self.energyvalueFocusLost)
        self.energyunitlabel = QLabel("MeV")

        self.chargelabel = QLabel("Charge")
        self.chargevalue = QLineEdit("0.2")
        self.chargevalue.setValidator(QDoubleValidator())
        self.chargevalue.editingFinished.connect(self.chargevalueFocusLost)
        self.chargeunitlabel = QLabel("nC")

        self.spreadlabel = QLabel("Gamma-spread")
        self.spreadvalue = QLineEdit("0.25")
        self.spreadvalue.setValidator(QDoubleValidator())
        self.spreadvalue.editingFinished.connect(self.spreadvalueFocusLost)
        self.chargeunitlabel1 = QLabel("%")

        self.elengthlabel = QLabel("Length")
        self.elengthvalue = QLineEdit("10")
        self.elengthvalue.setValidator(QDoubleValidator())
        self.elengthvalue.editingFinished.connect(self.elengthvalueFocusLost)
        self.elengthunitlabel = QLabel("ps")

        self.eemitxlabel = QLabel("X-emittance")
        self.eemitxvalue = QLineEdit("1")
        self.eemitxvalue.setValidator(QDoubleValidator())
        self.eemitxvalue.editingFinished.connect(self.eemitxvalueFocusLost)
        self.eemitxunitlabel = QLabel("mm*mrad")

        self.ebetaxlabel = QLabel("Beta x")
        self.ebetaxvalue = QLineEdit("20")
        self.ebetaxvalue.setValidator(QDoubleValidator())
        self.ebetaxvalue.editingFinished.connect(self.ebetaxvalueFocusLost)
        self.ebetaxunitlabel = QLabel("mm")

        self.eemitylabel = QLabel("Y-emittance")
        self.eemityvalue = QLineEdit("1")
        self.eemityvalue.setValidator(QDoubleValidator())
        self.eemityvalue.editingFinished.connect(self.eemityvalueFocusLost)
        self.eemityunitlabel = QLabel("mm*mrad")

        # Layout
        layout = QFormLayout()

        add_form_row_with_widgets(layout, self.energylabel, self.energyvalue, self.energyunitlabel)
        add_form_row_with_widgets(layout, self.chargelabel, self.chargevalue, self.chargeunitlabel)
        add_form_row_with_widgets(layout, self.spreadlabel, self.spreadvalue, self.chargeunitlabel1)
        add_form_row_with_widgets(layout, self.elengthlabel, self.elengthvalue, self.elengthunitlabel)
        add_form_row_with_widgets(layout, self.eemitxlabel, self.eemitxvalue, self.eemitxunitlabel)
        add_form_row_with_widgets(layout, self.ebetaxlabel, self.ebetaxvalue, self.ebetaxunitlabel)
        add_form_row_with_widgets(layout, self.eemitylabel, self.eemityvalue, self.eemityunitlabel)

        self.jPanel_el.setLayout(layout)

    def createLaserPulsePanel(self):
        self.jPanel_ph = QGroupBox("Laser pulse parameters")
        self.jPanel_ph.setMinimumSize(223, 350)

        # Create widgets
        self.phenergylabel = QLabel("Photon energy")
        self.phenergyvalue = QLineEdit("1.204")
        self.phenergyvalue.setValidator(QDoubleValidator())
        self.phenergyvalue.editingFinished.connect(self.phenergyvalueFocusLost)
        self.phenergyunitlabel = QLabel("eV")

        self.pulseenergylabel = QLabel("Pulse energy")
        self.pulseenergyvalue = QLineEdit("100")
        self.pulseenergyvalue.setValidator(QDoubleValidator())
        self.pulseenergyvalue.editingFinished.connect(self.pulseenergyvalueFocusLost)
        self.pulseenergyunitlabel = QLabel("mJ")

        self.puslelengthlabel = QLabel("Pulse length")
        self.pulselengthvalue = QLineEdit("10")
        self.pulselengthvalue.setValidator(QDoubleValidator())
        self.pulselengthvalue.editingFinished.connect(self.pulselengthvalueFocusLost)
        self.pulselengthunitlabel = QLabel("ps")

        self.pulserellabel = QLabel("Rayleigh length")
        self.pulserelvalue = QLineEdit("0.35")
        self.pulserelvalue.setValidator(QDoubleValidator())
        self.pulserelvalue.editingFinished.connect(self.pulserelvalueFocusLost)
        self.pulserelunitlable = QLabel("mm")

        self.pulsefreqlabel = QLabel("Pulse frequency")
        self.pulsefreqvalue = QLineEdit("1000")
        self.pulsefreqvalue.setValidator(QDoubleValidator())
        self.pulsefreqvalue.editingFinished.connect(self.pulsefreqvalueFocusLost)
        self.pulsefrequnitlabel = QLabel("Hz")

        self.pulsedelaylabel = QLabel("Delay")
        self.pulsedelayvalue = QLineEdit("0")
        self.pulsedelayvalue.setValidator(QDoubleValidator())
        self.pulsedelayvalue.editingFinished.connect(self.pulsedelayvalueFocusLost)
        self.pulsedelayunitlabel = QLabel("mm")

        self.ebetaylabel = QLabel("Beta y")
        self.ebetayvalue = QLineEdit("20")
        self.ebetayvalue.setValidator(QDoubleValidator())
        self.ebetayvalue.editingFinished.connect(self.ebetayvalueFocusLost)
        self.ebetayunitlabel = QLabel("mm")

        # Layout
        layout = QFormLayout()
        add_form_row_with_widgets(layout, self.phenergylabel, self.phenergyvalue, self.phenergyunitlabel)
        add_form_row_with_widgets(layout, self.pulseenergylabel, self.pulseenergyvalue, self.pulseenergyunitlabel)
        add_form_row_with_widgets(layout, self.puslelengthlabel, self.pulselengthvalue, self.pulselengthunitlabel)
        add_form_row_with_widgets(layout, self.pulserellabel, self.pulserelvalue, self.pulserelunitlable)
        add_form_row_with_widgets(layout, self.pulsefreqlabel, self.pulsefreqvalue, self.pulsefrequnitlabel)
        add_form_row_with_widgets(layout, self.pulsedelaylabel, self.pulsedelayvalue, self.pulsedelayunitlabel)
        add_form_row_with_widgets(layout, self.ebetaylabel, self.ebetayvalue, self.ebetayunitlabel)

        self.jPanel_ph.setLayout(layout)

    def createRelativePositionPanel(self):
        self.jPanel_sh = QGroupBox("Relative position")
        self.jPanel_sh.setMinimumSize(221, 180)

        # Create widgets
        self.eshiftxlabel = QLabel("X-shift")
        self.eshiftxvalue = QLineEdit("0")
        self.eshiftxvalue.setValidator(QDoubleValidator())
        self.eshiftxvalue.editingFinished.connect(self.eshiftxvalueFocusLost)
        self.eshiftxunitlabel = QLabel("mm")

        self.eshiftylabel = QLabel("Y-shift")
        self.eshiftyvalue = QLineEdit("0")
        self.eshiftyvalue.setValidator(QDoubleValidator())
        self.eshiftyvalue.editingFinished.connect(self.eshiftyvalueFocusLost)
        self.eshiftyunitlabel = QLabel("mm")

        self.eshiftzlabel = QLabel("Z-shift")
        self.eshiftzvalue = QLineEdit("0")
        self.eshiftzvalue.setValidator(QDoubleValidator())
        self.eshiftzvalue.editingFinished.connect(self.eshiftzvalueFocusLost)
        self.eshiftzunitlabel = QLabel("mm")

        self.pulseanglelabel = QLabel("Angle")
        self.pulseanglevalue = QLineEdit("52")
        self.pulseanglevalue.setValidator(QDoubleValidator())
        self.pulseanglevalue.editingFinished.connect(self.pulseanglevalueFocusLost)
        self.pulseangleunitlabel = QLabel("mrad")

        # Layout
        layout = QFormLayout()
        add_form_row_with_widgets(layout, self.eshiftxlabel, self.eshiftxvalue, self.eshiftxunitlabel)
        add_form_row_with_widgets(layout, self.eshiftylabel, self.eshiftyvalue, self.eshiftyunitlabel)
        add_form_row_with_widgets(layout, self.eshiftzlabel, self.eshiftzvalue, self.eshiftzunitlabel)
        add_form_row_with_widgets(layout, self.pulseanglelabel, self.pulseanglevalue, self.pulseangleunitlabel)

        self.jPanel_sh.setLayout(layout)

    def createExecutionPanel(self):
        self.jPanel_exec = QGroupBox("Execution")

        self.startbutton = QPushButton("Start")
        self.startbutton.clicked.connect(self.startbuttonActionPerformed)

        self.MainProgressBar = QProgressBar()

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.startbutton)
        layout.addWidget(self.MainProgressBar)

        self.jPanel_exec.setLayout(layout)
        print("Execution panel created")

    def createTabbedPane(self):
        self.jTabbedPane1 = QTabWidget()
        self.jTabbedPane1.setMinimumSize(713, 500)

        # Flux tab
        self.jPanel_xflux = QWidget()
        self.jPanel_xflux.setLayout(QHBoxLayout())

        self.jPanel_xflux_left = QWidget()
        self.jPanel_xflux_left.setMinimumSize(362, 318)
        self.jPanel_xflux.layout().addWidget(self.jPanel_xflux_left)

        self.jPanel_xflux_right = QWidget()
        self.jPanel_xflux_right.setMinimumSize(309, 318)
        self.jPanel_xflux.layout().addWidget(self.jPanel_xflux_right)

        self.jTabbedPane1.addTab(self.jPanel_xflux, "Flux")

        # Energy tab
        self.jPanel_xenergy = QWidget()
        self.jPanel_xenergy.setLayout(QHBoxLayout())

        self.jPanel_xenergy_left = QWidget()
        self.jPanel_xenergy_left.setMinimumSize(362, 318)
        self.jPanel_xenergy.layout().addWidget(self.jPanel_xenergy_left)

        self.jPanel_xenergy_right = QWidget()
        self.jPanel_xenergy_right.setMinimumSize(309, 318)
        self.jPanel_xenergy.layout().addWidget(self.jPanel_xenergy_right)

        self.jTabbedPane1.addTab(self.jPanel_xenergy, "Energy")

    def createSliderPanel(self):
        self.jPanel_slider = QWidget()

        self.jSlider_pickup = QSlider(Qt.Horizontal)
        self.jSlider_pickup.valueChanged.connect(self.jSlider_pickupStateChanged)

        self.totalFluxLabel = QLabel("Total flux:")
        self.totalFluxAngleLabel = QLabel("Within limits: ")

        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.jSlider_pickup)
        layout.addWidget(self.totalFluxLabel)
        layout.addWidget(self.totalFluxAngleLabel)

        self.jPanel_slider.setLayout(layout)

    def createMenuBar(self):
        self.jMenuBarMain = QMenuBar()

        # File menu
        self.jMenuFile = QMenu("File")

        self.jMenuItemSaveParam = QAction("Save parameters", self)
        self.jMenuItemSaveParam.triggered.connect(self.jMenuItemSaveParamActionPerformed)
        self.jMenuFile.addAction(self.jMenuItemSaveParam)

        self.jMenuItemLoadParam = QAction("Load parameters", self)
        self.jMenuItemLoadParam.triggered.connect(self.jMenuItemLoadParamActionPerformed)
        self.jMenuFile.addAction(self.jMenuItemLoadParam)

        self.jMenuFile.addSeparator()

        self.jMenuItemExit = QAction("Exit", self)
        self.jMenuItemExit.triggered.connect(self.jMenuItemExitActionPerformed)
        self.jMenuFile.addAction(self.jMenuItemExit)

        self.jMenuBarMain.addMenu(self.jMenuFile)

        # Calculations menu
        self.jMenuCalc = QMenu("Calculations")

        self.jMenuItemBrilliance = QAction("Brilliance...", self)
        self.jMenuItemBrilliance.triggered.connect(self.jMenuItemBrillianceActionPerformed)
        self.jMenuCalc.addAction(self.jMenuItemBrilliance)

        self.jMenuItemGeometricFactor = QAction("Full flux/Geometric factor...", self)
        self.jMenuItemGeometricFactor.triggered.connect(self.jMenuItemGeometricFactorActionPerformed)
        self.jMenuCalc.addAction(self.jMenuItemGeometricFactor)

        self.jMenuItemPolarization = QAction("Polarization...", self)
        self.jMenuItemPolarization.triggered.connect(self.jMenuItemPolarizationActionPerformed)
        self.jMenuCalc.addAction(self.jMenuItemPolarization)

        self.jMenuBarMain.addMenu(self.jMenuCalc)

        # Shadow menu
        self.jMenuShadow = QMenu("Shadow")

        self.jMenuItemSource = QAction("Generate source...", self)
        self.jMenuItemSource.triggered.connect(self.jMenuItemSourceActionPerformed)
        self.jMenuShadow.addAction(self.jMenuItemSource)

        self.jMenuItemSourceParam = QAction("Parameters...", self)
        self.jMenuItemSourceParam.triggered.connect(self.jMenuItemSourceParamActionPerformed)
        self.jMenuShadow.addAction(self.jMenuItemSourceParam)

        # Polarization submenu
        self.jMenuPolarization = QMenu("Polarization...")

        # Используем QActionGroup вместо QButtonGroup
        self.polarizationActionGroup = QActionGroup(self)
        self.polarizationActionGroup.setExclusive(True)

        # Unpolarized
        self.jRadioButtonMenuItemUnPolarized = QAction("Unpolarized", self)
        self.jRadioButtonMenuItemUnPolarized.setCheckable(True)
        self.polarizationActionGroup.addAction(self.jRadioButtonMenuItemUnPolarized)
        self.jRadioButtonMenuItemUnPolarized.triggered.connect(self.jRadioButtonMenuItemUnPolarizedItemStateChanged)
        self.jMenuPolarization.addAction(self.jRadioButtonMenuItemUnPolarized)

        # Linear
        self.jRadioButtonMenuItemLinearPolarized = QAction("Linear", self)
        self.jRadioButtonMenuItemLinearPolarized.setCheckable(True)
        self.jRadioButtonMenuItemLinearPolarized.setChecked(True)
        self.polarizationActionGroup.addAction(self.jRadioButtonMenuItemLinearPolarized)
        self.jRadioButtonMenuItemLinearPolarized.triggered.connect(
            self.jRadioButtonMenuItemLinearPolarizedItemStateChanged)
        self.jMenuPolarization.addAction(self.jRadioButtonMenuItemLinearPolarized)

        # Circular
        self.jRadioButtonMenuItemCircularPolarized = QAction("Circular", self)
        self.jRadioButtonMenuItemCircularPolarized.setCheckable(True)
        self.polarizationActionGroup.addAction(self.jRadioButtonMenuItemCircularPolarized)
        self.jRadioButtonMenuItemCircularPolarized.triggered.connect(
            self.jRadioButtonMenuItemCircularPolarizedItemStateChanged)
        self.jMenuPolarization.addAction(self.jRadioButtonMenuItemCircularPolarized)

        # Automatic
        self.jRadioButtonMenuItemAutoPolarized = QAction("Automatic", self)
        self.jRadioButtonMenuItemAutoPolarized.setCheckable(True)
        self.polarizationActionGroup.addAction(self.jRadioButtonMenuItemAutoPolarized)
        self.jRadioButtonMenuItemAutoPolarized.triggered.connect(self.jRadioButtonMenuItemAutoPolarizedItemStateChanged)
        self.jMenuPolarization.addAction(self.jRadioButtonMenuItemAutoPolarized)

        self.jMenuShadow.addMenu(self.jMenuPolarization)

        self.jMenuShadow.addSeparator()

        self.jMenuItemConv = QAction("Converter...", self)
        self.jMenuItemConv.triggered.connect(self.jMenuItemConvActionPerformed)
        self.jMenuShadow.addAction(self.jMenuItemConv)

        self.jMenuBarMain.addMenu(self.jMenuShadow)

        # Options menu
        self.jMenuOptions = QMenu("Options")

        self.jMenuItemLaserPolarization = QAction("Laser polarization...", self)
        self.jMenuItemLaserPolarization.triggered.connect(self.jMenuItemLaserPolarizationActionPerformed)
        self.jMenuOptions.addAction(self.jMenuItemLaserPolarization)

        self.jMenuOptions.addSeparator()

        self.jMenuItemSize = QAction("Graphical parameters...", self)
        self.jMenuItemSize.triggered.connect(self.jMenuItemSizeActionPerformed)
        self.jMenuOptions.addAction(self.jMenuItemSize)

        self.jMenuItemNumerical = QAction("Numerical parameters...", self)
        self.jMenuItemNumerical.triggered.connect(self.jMenuItemNumericalActionPerformed)
        self.jMenuOptions.addAction(self.jMenuItemNumerical)

        self.jMenuOptions.addSeparator()

        self.jCheckBoxMenuItemSpread = QAction("Velocity spread", self)
        self.jCheckBoxMenuItemSpread.setCheckable(True)
        self.jCheckBoxMenuItemSpread.triggered.connect(self.jCheckBoxMenuItemSpreadActionPerformed)
        self.jMenuOptions.addAction(self.jCheckBoxMenuItemSpread)

        self.jCheckBoxMenuItemMonteCarlo = QAction("MonteCarlo", self)
        self.jCheckBoxMenuItemMonteCarlo.setCheckable(True)
        self.jCheckBoxMenuItemMonteCarlo.setChecked(True)
        self.jCheckBoxMenuItemMonteCarlo.triggered.connect(self.jCheckBoxMenuItemMonteCarloActionPerformed)
        self.jMenuOptions.addAction(self.jCheckBoxMenuItemMonteCarlo)

        self.jCheckBoxMenuItemCompton = QAction("Compton", self)
        self.jCheckBoxMenuItemCompton.setCheckable(True)
        self.jCheckBoxMenuItemCompton.setChecked(True)
        self.jCheckBoxMenuItemCompton.triggered.connect(self.jCheckBoxMenuItemComptonActionPerformed)
        self.jMenuOptions.addAction(self.jCheckBoxMenuItemCompton)

        self.jMenuOptions.addSeparator()

        # Look&Feel submenu
        self.jMenuSkin = QMenu("Look&Feel...")

        # Используем QActionGroup вместо QButtonGroup
        self.skinActionGroup = QActionGroup(self)
        self.skinActionGroup.setExclusive(True)

        # Default
        self.jRadioButtonMenuDefault = QAction("Default", self)
        self.jRadioButtonMenuDefault.setCheckable(True)
        self.skinActionGroup.addAction(self.jRadioButtonMenuDefault)
        self.jRadioButtonMenuDefault.triggered.connect(self.jRadioButtonMenuDefaultItemStateChanged)
        self.jMenuSkin.addAction(self.jRadioButtonMenuDefault)

        # System
        self.jRadioButtonMenuSystem = QAction("System", self)
        self.jRadioButtonMenuSystem.setCheckable(True)
        self.skinActionGroup.addAction(self.jRadioButtonMenuSystem)
        self.jRadioButtonMenuSystem.triggered.connect(self.jRadioButtonMenuSystemItemStateChanged)
        self.jMenuSkin.addAction(self.jRadioButtonMenuSystem)

        # Nimbus
        self.jRadioButtonMenuNimbus = QAction("Nimbus", self)
        self.jRadioButtonMenuNimbus.setCheckable(True)
        self.jRadioButtonMenuNimbus.setChecked(True)
        self.skinActionGroup.addAction(self.jRadioButtonMenuNimbus)
        self.jRadioButtonMenuNimbus.triggered.connect(self.jRadioButtonMenuNimbusItemStateChanged)
        self.jMenuSkin.addAction(self.jRadioButtonMenuNimbus)

        # Добавление подменю Skin в меню Options
        self.jMenuOptions.addMenu(self.jMenuSkin)

        # Добавление Options и Help в главное меню
        self.jMenuBarMain.addMenu(self.jMenuOptions)

        # Help menu
        self.jMenuHelp = QMenu("Help")

        self.HelpItem = QAction("Contents...", self)
        self.HelpItem.triggered.connect(self.HelpItemActionPerformed)
        self.jMenuHelp.addAction(self.HelpItem)

        self.jMenuItemAbout = QAction("About TSourceXG", self)
        self.jMenuItemAbout.triggered.connect(self.jMenuItemAboutActionPerformed)
        self.jMenuHelp.addAction(self.jMenuItemAbout)

        self.jMenuBarMain.addMenu(self.jMenuHelp)

        self.setMenuBar(self.jMenuBarMain)

    # Event handler methods
    def BrillianceCalcBoxActionPerformed(self, evt):
        pass

    def BrillianceCalcStartActionPerformed(self, evt):
        pass

    def BrillianceCalcSaveActionPerformed(self, evt):
        pass

    def BrilminvalueFocusLost(self, evt):
        pass

    def BrilminvalueActionPerformed(self, evt):
        pass

    def BrilmaxvalueFocusLost(self, evt):
        pass

    def BrilmaxvalueActionPerformed(self, evt):
        pass

    def jCheckBoxSpreadActionPerformed(self, evt):
        pass

    def angleValueFocusLost(self, evt):
        pass

    def angleValueActionPerformed(self, evt):
        pass

    def energyValueFocusLost(self, evt):
        pass

    def energyValueActionPerformed(self, evt):
        pass

    def GFCalcBoxActionPerformed(self, evt):
        pass

    def GFCalcStartActionPerformed(self, evt):
        pass

    def GFCalcSaveActionPerformed(self, evt):
        pass

    def GFminvalueFocusLost(self, evt):
        pass

    def GFminvalueActionPerformed(self, evt):
        pass

    def GFmaxvalueFocusLost(self, evt):
        pass

    def GFmaxvalueActionPerformed(self, evt):
        pass

    def GFValueSelectionBoxActionPerformed(self, evt):
        pass

    def polarizationCalcBoxActionPerformed(self, evt):
        pass

    def polarizationCalcStartActionPerformed(self, evt):
        pass

    def polarizationCalcSaveActionPerformed(self, evt):
        pass

    def polminvalueFocusLost(self, evt):
        pass

    def polminvalueActionPerformed(self, evt):
        pass

    def polmaxvalueFocusLost(self, evt):
        pass

    def polmaxvalueActionPerformed(self, evt):
        pass

    def jPolCheckBoxSpreadActionPerformed(self, evt):
        pass

    def polAngleValueFocusLost(self, evt):
        pass

    def polAngleValueActionPerformed(self, evt):
        pass

    def polEnergyValueFocusLost(self, evt):
        pass

    def polEnergyValueActionPerformed(self, evt):
        pass

    def jRayStopButtonActionPerformed(self, evt):
        pass

    def formMouseMoved(self, evt):
        pass

    def energyvalueFocusLost(self, evt):
        pass

    def energyvalueActionPerformed(self, evt):
        pass

    def chargevalueFocusLost(self, evt):
        pass

    def chargevalueActionPerformed(self, evt):
        pass

    def spreadvalueFocusLost(self, evt):
        pass

    def spreadvalueActionPerformed(self, evt):
        pass

    def elengthvalueFocusLost(self, evt):
        pass

    def elengthvalueActionPerformed(self, evt):
        pass

    def eemitxvalueFocusLost(self, evt):
        pass

    def eemitxvalueActionPerformed(self, evt):
        pass

    def ebetaxvalueFocusLost(self, evt):
        pass

    def ebetaxvalueActionPerformed(self, evt):
        pass

    def eemityvalueFocusLost(self, evt):
        pass

    def eemityvalueActionPerformed(self, evt):
        pass

    def phenergyvalueFocusLost(self, evt):
        pass

    def phenergyvalueActionPerformed(self, evt):
        pass

    def pulseenergyvalueFocusLost(self, evt):
        pass

    def pulseenergyvalueActionPerformed(self, evt):
        pass

    def pulselengthvalueFocusLost(self, evt):
        pass

    def pulselengthvalueActionPerformed(self, evt):
        pass

    def pulserelvalueFocusLost(self, evt):
        pass

    def pulserelvalueActionPerformed(self, evt):
        pass

    def pulsefreqvalueFocusLost(self, evt):
        pass

    def pulsefreqvalueActionPerformed(self, evt):
        pass

    def pulsedelayvalueFocusLost(self, evt):
        pass

    def pulsedelayvalueActionPerformed(self, evt):
        pass

    def ebetayvalueFocusLost(self, evt):
        pass

    def ebetayvalueActionPerformed(self, evt):
        pass

    def startbuttonActionPerformed(self, evt):
        if self.working:
            if self.mainWorker and self.mainWorker.isRunning():
                self.mainWorker.terminate()
            self.working = False
            self.startbutton.setText("Start")
            return

        self.working = True
        self.startbutton.setText("Stop")
        self.MainProgressBar.setValue(0)

        # Update parameters from UI
        self.updateParametersFromUI()

        # Start calculation in a separate thread
        self.mainWorker = CalculationThread(self)
        self.mainWorker.progress_updated.connect(self.updateProgress)
        self.mainWorker.calculation_finished.connect(self.onCalculationFinished)
        self.mainWorker.start()

    def updateParametersFromUI(self):
        # Update electron parameters
        self.ebunch.gamma = float(self.energyvalue.text()) / 0.5109989461
        self.ebunch.number = float(self.chargevalue.text()) / 1.602E-19 * 1.0E-9
        self.ebunch.delgamma = float(self.spreadvalue.text()) / 200.0
        self.ebunch.length = float(self.elengthvalue.text()) * 3.0E-4 / 2.0
        self.ebunch.epsx = float(self.eemitxvalue.text()) * 1e-6
        self.ebunch.epsy = float(self.eemityvalue.text()) * 1e-6
        self.ebunch.betax = float(self.ebetaxvalue.text()) * 1e-3
        self.ebunch.betay = float(self.ebetayvalue.text()) * 1e-3

        # Update shift
        self.ebunch.shift = Vector(
            float(self.eshiftxvalue.text()) * 1e-3,
            float(self.eshiftyvalue.text()) * 1e-3,
            float(self.eshiftzvalue.text()) * 1e-3
        )

        # Update laser parameters
        self.lpulse.photon_energy = float(self.phenergyvalue.text()) * 1.602E-19
        self.lpulse.pulse_energy = float(self.pulseenergyvalue.text()) * 0.001
        self.lpulse.length = float(self.pulselengthvalue.text()) * 3.0E-4 / 2.0
        self.lpulse.rlength = float(self.pulserelvalue.text()) * 1e-3
        self.lpulse.fq = float(self.pulsefreqvalue.text())
        self.lpulse.delay = float(self.pulsedelayvalue.text()) * 1e-3

        # Update laser direction
        angle = float(self.pulseanglevalue.text()) * 1e-3
        self.lpulse.direction = Vector(0, math.sin(angle), math.cos(angle))

        # Recreate ThompsonSource with updated parameters
        self.tsource = ThompsonSource(self.lpulse, self.ebunch)

    def jSlider_pickupStateChanged(self, evt):
        self.sliderposition = self.jSlider_pickup.value()
        self.hoffset = self.xsize * self.xstep * (self.sliderposition - 50) / 100.0

        if not self.working:
            self.startbuttonActionPerformed(None)

    def eshiftxvalueFocusLost(self, evt):
        pass

    def eshiftxvalueActionPerformed(self, evt):
        pass

    def eshiftyvalueFocusLost(self, evt):
        pass

    def eshiftyvalueActionPerformed(self, evt):
        pass

    def eshiftzvalueFocusLost(self, evt):
        pass

    def eshiftzvalueActionPerformed(self, evt):
        pass

    def pulseanglevalueFocusLost(self, evt):
        pass

    def pulseanglevalueActionPerformed(self, evt):
        pass

    def jMenuItemSaveParamActionPerformed(self, evt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Parameters",
            "",
            "INI Files (*.ini);;All Files (*)",
            options=options
        )

        if file_name:
            self.saveParameters(file_name)

    def jMenuItemLoadParamActionPerformed(self, evt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Load Parameters",
            "",
            "INI Files (*.ini);;All Files (*)",
            options=options
        )

        if file_name:
            self.loadParameters(file_name)

    def jMenuItemExitActionPerformed(self, evt):
        QApplication.quit()

    def jMenuItemBrillianceActionPerformed(self, evt):
        self.brillianceCalc.show()

    def jMenuItemGeometricFactorActionPerformed(self, evt):
        self.gfCalc.show()

    def jMenuItemPolarizationActionPerformed(self, evt):
        self.polarizationCalc.show()

    def jMenuItemSourceActionPerformed(self, evt):
        pass

    def jMenuItemSourceParamActionPerformed(self, evt):
        pass

    def jRadioButtonMenuItemUnPolarizedItemStateChanged(self, evt):
        pass

    def jRadioButtonMenuItemLinearPolarizedItemStateChanged(self, evt):
        pass

    def jRadioButtonMenuItemCircularPolarizedItemStateChanged(self, evt):
        pass

    def jRadioButtonMenuItemAutoPolarizedItemStateChanged(self, evt):
        pass

    def jMenuItemConvActionPerformed(self, evt):
        pass

    def jMenuItemLaserPolarizationActionPerformed(self, evt):
        pass

    def jMenuItemSizeActionPerformed(self, evt):
        dialog = QDialog(self)
        dialog.setWindowTitle("Graphical Parameters")
        layout = QVBoxLayout()

        # Add controls for graphical parameters here
        # ...

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)

        dialog.setLayout(layout)
        if dialog.exec_() == QDialog.Accepted:
            # Apply new graphical parameters
            pass

    def jMenuItemNumericalActionPerformed(self, evt):
        pass

    def jCheckBoxMenuItemSpreadActionPerformed(self, evt):
        pass

    def jCheckBoxMenuItemMonteCarloActionPerformed(self, evt):
        pass

    def jCheckBoxMenuItemComptonActionPerformed(self, evt):
        pass

    def jRadioButtonMenuDefaultItemStateChanged(self, evt):
        pass

    def jRadioButtonMenuSystemItemStateChanged(self, evt):
        pass

    def jRadioButtonMenuNimbusItemStateChanged(self, evt):
        pass

    def HelpItemActionPerformed(self, evt):
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Help")
        help_dialog.resize(600, 400)

        scroll = QScrollArea()
        text = QTextBrowser()
        text.setOpenExternalLinks(True)

        try:
            # Load help from HTML file if available
            help_file = os.path.join(os.path.dirname(__file__), "help", "thomson_help.html")
            if os.path.exists(help_file):
                with open(help_file, 'r') as f:
                    text.setHtml(f.read())
            else:
                text.setHtml("<h1>Help</h1><p>Help file not found.</p>")
        except Exception as e:
            text.setHtml(f"<h1>Error</h1><p>Could not load help: {str(e)}</p>")

        scroll.setWidget(text)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout()
        layout.addWidget(scroll)
        help_dialog.setLayout(layout)
        help_dialog.exec_()

    def jMenuItemAboutActionPerformed(self, evt):
        about_text = """
        <h2>Thomson Scattering Simulation</h2>
        <p>Version: 1.0</p>
        <p>Build date: {}</p>
        <p>Author: Your Name</p>
        <p>This software simulates Thomson scattering processes.</p>
        """.format(datetime.now().strftime("%Y-%m-%d"))

        QMessageBox.about(self, "About Thomson Simulation", about_text)

    def updateProgress(self, value):
        self.MainProgressBar.setValue(value)

    def onCalculationFinished(self):
        self.working = False
        self.startbutton.setText("Start")
        self.drawCharts()

    def drawCharts(self):
        # Clear the figure
        self.figure.clear()

        # Determine the view type
        view_type = "Color Chart"  # Default, should be replaced with actual combo box value

        if view_type == "Color Chart":
            # Draw color map
            ax = self.figure.add_subplot(111)
            x = np.linspace(0, self.xsize * self.xstep, self.xsize)
            y = np.linspace(0, self.ysize * self.ystep, self.ysize)
            X, Y = np.meshgrid(x, y)

            # Get data for the plot
            Z = np.zeros((self.ysize, self.xsize))
            for i in range(self.xsize):
                for j in range(self.ysize):
                    Z[j, i] = self.fluxdata.func(X[j, i], Y[j, i])

            # Create color map
            cmap = LinearSegmentedColormap.from_list("jet", ["blue", "cyan", "yellow", "red"])
            im = ax.imshow(Z, cmap=cmap, extent=[0, self.xsize * self.xstep, 0, self.ysize * self.ystep],
                           origin='lower', aspect='auto')

            # Add color bar
            self.figure.colorbar(im, ax=ax, label='Flux (mrad⁻²·s⁻¹·10⁷)')
            ax.set_xlabel('theta_x (mrad)')
            ax.set_ylabel('theta_y (mrad)')
            ax.set_title('Thomson Scattering Flux Distribution')

        elif view_type == "Linear Scale":
            # Draw linear plot
            ax = self.figure.add_subplot(111)
            x = np.linspace(0, self.xsize * self.xstep, self.xsize)
            y = [self.xenergycrossdata.func(xi) for xi in x]
            ax.plot(x, y)
            ax.set_xlabel('theta_y (mrad)')
            ax.set_ylabel('Energy (keV)')
            ax.set_title('X-ray Energy Cross Section')

        # Redraw the canvas
        self.canvas.draw()


class CalculationThread(QThread):
    progress_updated = pyqtSignal(int)
    calculation_finished = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def run(self):
        try:
            # Calculate total flux
            self.parent.tsource.calculate_total_flux()
            self.progress_updated.emit(25)

            # Calculate geometric factor
            self.parent.tsource.calculate_geometric_factor()
            self.progress_updated.emit(50)

            # Setup data for charts
            self.parent.fluxdata.setup(
                self.parent.xsize,
                self.parent.ysize,
                self.parent.xstep,
                self.parent.ystep,
                0.0,
                0.0
            )
            self.progress_updated.emit(75)

            # Setup cross section data
            if self.parent.fluxdata.udata is not None:
                self.parent.xenergycrossdata.setup_from_data(
                    np.array(self.parent.fluxdata.udata),
                    (self.parent.xsize - 1) * self.parent.sliderposition // 100,
                    False,
                    self.parent.ysize,
                    self.parent.ystep,
                    -self.parent.ystep * self.parent.ysize / 2.0
                )

            self.progress_updated.emit(100)

        except Exception as e:
            print(f"Calculation error: {str(e)}")
        finally:
            self.calculation_finished.emit()


def add_form_row_with_widgets(form_layout, label_widget, *field_widgets):
    """
    Добавляет строку в QFormLayout, где слева — QLabel или QWidget,
    а справа — несколько виджетов, размещённых горизонтально.
    """
    hbox = QHBoxLayout()
    for widget in field_widgets:
        hbox.addWidget(widget)

    container = QWidget()
    container.setLayout(hbox)
    form_layout.addRow(label_widget, container)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    print("Окно должно отображаться")
    window = ThomsonJFrame()
    print("Окно должно отображаться")
    window.show()
    print("Окно должно отображаться")
    sys.exit(app.exec_())