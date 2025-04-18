import math
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from copy import deepcopy


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

    def normalize(self) -> None:
        norm = np.linalg.norm(self._data)
        if norm > 0:
            self._data /= norm

    def copy(self) -> 'Vector':
        return Vector(self.x, self.y, self.z)


class LaserPulse:
    """Класс для описания лазерного импульса"""

    # Физическая константа
    HC: float = 3.1614e-26  # Произведение постоянной Планка и скорости света (Дж·м)

    def __init__(self):
        """Инициализация с параметрами по умолчанию"""
        self._photon_energy: float = 1.928808e-19  # Энергия фотона (Дж)
        self._number: float = 0.0  # Количество фотонов
        self._length: float = 0.0015  # Длина импульса (м)
        self._direction: Vector = Vector(0.0, 0.0, 1.0)  # Направление импульса
        self._rlength: float = 3.5e-4  # Рэлеевская длина (м)
        self._fq: float = 1000.0  # Частота повторения (Гц)
        self._delay: float = 0.0  # Задержка (с)
        self._polarization: List[float] = [1.0, 0.0, 0.0]  # Вектор поляризации [ksi1, ksi2, ksi3]
        self._rk: float = self.HC / self._photon_energy  # Вспомогательный параметр

        # Инициализация направления с небольшим наклоном (~3 градуса)
        self._direction.set(2, math.cos(0.052))
        self._direction.set(1, math.sin(0.052))
        self._direction.normalize()

        # Установка энергии импульса по умолчанию (0.1 Дж)
        self.pulse_energy = 0.1 / self._photon_energy


    def copy(self) -> 'LaserPulse':
        """Создание глубокой копии объекта"""
        new_pulse = LaserPulse()
        new_pulse._photon_energy = self._photon_energy
        new_pulse._number = self._number
        new_pulse._length = self._length
        new_pulse._direction = self._direction.copy()
        new_pulse._rlength = self._rlength
        new_pulse._fq = self._fq
        new_pulse._delay = self._delay
        new_pulse._polarization = self._polarization.copy()
        new_pulse._rk = self._rk
        return new_pulse

    # ------------------------------------------
    # Основные свойства и методы
    # ------------------------------------------

    @property
    def photon_energy(self) -> float:
        """Энергия одного фотона (Дж)"""
        return self._photon_energy

    @photon_energy.setter
    def photon_energy(self, value: float) -> None:
        self._photon_energy = value
        self._rk = self.HC / value

    @property
    def photon_number(self) -> float:
        """Количество фотонов в импульсе"""
        return self._number

    @photon_number.setter
    def photon_number(self, value: float) -> None:
        self._number = value

    @property
    def pulse_energy(self) -> float:
        """Полная энергия импульса (Дж)"""
        return self._number * self._photon_energy

    @pulse_energy.setter
    def pulse_energy(self, value: float) -> None:
        """Установка энергии импульса через полную энергию"""
        self._number = value / self._photon_energy

    @property
    def length(self) -> float:
        """Длина импульса (м)"""
        return self._length

    @length.setter
    def length(self, value: float) -> None:
        self._length = value

    @property
    def direction(self) -> Vector:
        """Вектор направления импульса"""
        return self._direction

    @direction.setter
    def direction(self, value: Vector) -> None:
        self._direction = value.copy()
        self._direction.normalize()

    @property
    def rlength(self) -> float:
        """Рэлеевская длина (м)"""
        return self._rlength

    @rlength.setter
    def rlength(self, value: float) -> None:
        self._rlength = value

    @property
    def fq(self) -> float:
        """Частота повторения импульсов (Гц)"""
        return self._fq

    @fq.setter
    def fq(self, value: float) -> None:
        self._fq = value

    @property
    def delay(self) -> float:
        """Задержка импульса (с)"""
        return self._delay

    @delay.setter
    def delay(self, value: float) -> None:
        self._delay = value

    @property
    def polarization(self) -> Tuple[float, float, float]:
        """Вектор поляризации (ksi1, ksi2, ksi3)"""
        return tuple(self._polarization)

    def set_polarization(self, ksi1: float, ksi2: float, ksi3: float) -> None:
        """Установка вектора поляризации"""
        self._polarization = [ksi1, ksi2, ksi3]
        # Нормализация не требуется, так как это параметры Стокса

    # ------------------------------------------
    # Методы для расчетов
    # ------------------------------------------

    def get_width(self, z: float) -> float:
        """Ширина пучка на расстоянии z (м)"""
        return math.sqrt((self._rlength + z ** 2 / self._rlength) * self._rk)

    def set_width(self, width: float) -> None:
        """Установка ширины пучка"""
        self._rlength = width ** 2 / self._rk

    def get_width_squared(self, z: float) -> float:
        """Квадрат ширины пучка на расстоянии z (м²)"""
        return (self._rlength + z ** 2 / self._rlength) * self._rk

    def __repr__(self) -> str:
        return (f"LaserPulse(energy={self.pulse_energy:.3e} J, "
                f"photons={self.photon_number:.3e}, "
                f"length={self.length:.3e} m)")


# Пример использования
if __name__ == "__main__":
    pulse = LaserPulse()

    # Изменение параметров
    pulse.photon_energy = 2.0e-19
    pulse.pulse_energy = 0.5  # 0.5 Дж
    pulse.direction = Vector(0.1, 0.1, 0.99)
    pulse.set_polarization(0.8, 0.5, 0.3)

    # Расчеты
    width = pulse.get_width(z=0.1)
    print(f"Ширина пучка на расстоянии 10 см: {width:.3e} м")

    # Копирование
    pulse_copy = pulse.copy()