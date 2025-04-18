import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Vector:
    """Реализация 3D-вектора (замена org.la4j.Vector)"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self):
        self._data = np.array([self.x, self.y, self.z])

    def get(self, index: int) -> float:
        return self._data[index]

    def set(self, index: int, value: float) -> None:
        self._data[index] = value

    def copy(self) -> 'Vector':
        return Vector(self.x, self.y, self.z)


class ElectronBunch:
    """Класс для работы с электронным пучком"""

    # Константы (аналоги static final в Java)
    E: float = 1.602e-19  # Заряд электрона (Кл)
    MC2: float = 0.5109989461  # Энергия покоя (МэВ)

    def __init__(self):
        """Инициализация с параметрами по умолчанию"""
        self._shift = Vector()
        self._gamma = 97.84755992474248
        self._number = 1.2484394506866417e9
        self._delgamma = 0.00125
        self._length = 0.0015
        self._epsx = 1.0e-6
        self._epsy = 1.0e-6
        self._betax = 0.02
        self._betay = 0.02

    def copy(self) -> 'ElectronBunch':
        """Создание копии объекта (аналог clone() в Java)"""
        new_bunch = ElectronBunch()
        new_bunch._shift = self._shift.copy()
        new_bunch._gamma = self._gamma
        new_bunch._number = self._number
        new_bunch._delgamma = self._delgamma
        new_bunch._length = self._length
        new_bunch._epsx = self._epsx
        new_bunch._epsy = self._epsy
        new_bunch._betax = self._betax
        new_bunch._betay = self._betay
        return new_bunch

    # ------------------------------------------
    # Свойства (геттеры и сеттеры)
    # ------------------------------------------
    @property
    def shift(self) -> Vector:
        return self._shift

    @shift.setter
    def shift(self, value: Vector) -> None:
        self._shift = value

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self._gamma = value

    @property
    def number(self) -> float:
        return self._number

    @number.setter
    def number(self, value: float) -> None:
        self._number = value

    @property
    def delgamma(self) -> float:
        return self._delgamma

    @delgamma.setter
    def delgamma(self, value: float) -> None:
        self._delgamma = value

    @property
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, value: float) -> None:
        self._length = value

    @property
    def epsx(self) -> float:
        return self._epsx

    @epsx.setter
    def epsx(self, value: float) -> None:
        self._epsx = value

    @property
    def epsy(self) -> float:
        return self._epsy

    @epsy.setter
    def epsy(self, value: float) -> None:
        self._epsy = value

    @property
    def betax(self) -> float:
        return self._betax

    @betax.setter
    def betax(self, value: float) -> None:
        self._betax = value

    @property
    def betay(self) -> float:
        return self._betay

    @betay.setter
    def betay(self, value: float) -> None:
        self._betay = value

    # ------------------------------------------
    # Методы для расчетов
    # ------------------------------------------
    def get_x_width(self, z: float) -> float:
        """Ширина пучка по X на расстоянии z"""
        return math.sqrt((self._betax + z ** 2 / self._betax) * self._epsx / self._gamma)

    def set_x_width(self, width: float) -> None:
        """Установка ширины пучка по X"""
        self._betax = width ** 2 / self._epsx * self._gamma

    def get_x_width_squared(self, z: float) -> float:
        """Квадрат ширины пучка по X"""
        return (self._betax + z ** 2 / self._betax) * self._epsx / self._gamma

    def get_x_spread(self) -> float:
        """Разброс по X"""
        return math.sqrt(self._epsx / self._gamma / self._betax)

    def get_y_width(self, z: float) -> float:
        """Ширина пучка по Y на расстоянии z"""
        return math.sqrt((self._betay + z ** 2 / self._betay) * self._epsy / self._gamma)

    def set_y_width(self, width: float) -> None:
        """Установка ширины пучка по Y"""
        self._betay = width ** 2 / self._epsy * self._gamma

    def get_y_width_squared(self, z: float) -> float:
        """Квадрат ширины пучка по Y"""
        return (self._betay + z ** 2 / self._betay) * self._epsy / self._gamma

    def get_y_spread(self) -> float:
        """Разброс по Y"""
        return math.sqrt(self._epsy / self._gamma / self._betay)

    def get_width(self, z: float) -> float:
        """Общая ширина пучка"""
        return math.sqrt(self.get_x_width(z) * self.get_y_width(z))

    def get_width_squared(self, z: float) -> float:
        """Квадрат общей ширины пучка"""
        return math.sqrt(self.get_x_width_squared(z) * self.get_y_width_squared(z))

    def get_spread(self) -> float:
        """Общий разброс"""
        return math.sqrt(self.get_x_spread() * self.get_y_spread())

    def angle_x_distribution(self, theta_x: float) -> float:
        """Распределение по углу theta_x"""
        dpx = self.get_x_spread()
        return math.exp(-(theta_x / dpx) ** 2) / dpx / math.sqrt(math.pi)

    def angle_y_distribution(self, theta_y: float) -> float:
        """Распределение по углу theta_y"""
        dpy = self.get_y_spread()
        return math.exp(-(theta_y / dpy) ** 2) / dpy / math.sqrt(math.pi)

    def angle_distribution(self, theta_x: float, theta_y: float) -> float:
        """Общее угловое распределение"""
        dpx = self.get_x_spread()
        dpy = self.get_y_spread()
        exponent = -((theta_x / dpx) ** 2 + (theta_y / dpy) ** 2)
        return math.exp(exponent) / (dpx * dpy * math.pi)