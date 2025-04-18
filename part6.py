import math
import numpy as np
import threading
import concurrent.futures
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
from scipy.integrate import quad
from scipy.integrate._quadrature import romb  # Исправленный импорт для romberg
from scipy.special import erf
import cmath
import random

# Подключение ваших классов
from part1 import ColorChartParam
from part2 import ElectronBunch
from part3 import JetPaintScale
from part4 import LaserPulse
from part5 import LinearChartParam
# Вспомогательные классы
@dataclass
class Vector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self):
        self._data = np.array([self.x, self.y, self.z])

    def inner_product(self, other: 'Vector') -> float:
        return np.dot(self._data, other._data)

    def norm(self) -> float:
        return np.linalg.norm(self._data)

    def normalize(self) -> 'Vector':
        n = self.norm()
        return Vector(*(self._data / n)) if n > 0 else self.copy()

    def cross(self, other: 'Vector') -> 'Vector':
        return Vector(*np.cross(self._data, other._data))

    def copy(self) -> 'Vector':
        return Vector(self.x, self.y, self.z)

    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(*(self._data * scalar))

    def __truediv__(self, scalar: float) -> 'Vector':
        return Vector(*(self._data / scalar))

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(*(self._data + other._data))

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(*(self._data - other._data))

    def outer_product(self, other: 'Vector') -> np.ndarray:
        return np.outer(self._data, other._data)

    def to_array(self) -> np.ndarray:
        return self._data.copy()


class ThompsonSource:
    """Полная реализация источника Томсона/Комптона на Python"""

    # Физические константы
    SIGMA_T: float = 6.65e-29  # Сечение Томсона [m^2]
    HC: float = 3.1614e-26  # h*c [J*m]
    MC2: float = 0.5109989461e6  # Энергия покоя электрона [eV]
    NUMBER_OF_COLUMNS: int = 18
    NUMBER_OF_POL_PARAM: int = 4
    MAXIMAL_NUMBER_OF_EVALUATIONS: int = 1000000
    INT_RANGE: float = 3.0
    SHIFT: float = 1.0e10

    def __init__(self, laser_pulse: LaserPulse, electron_bunch: ElectronBunch):
        self.lp = laser_pulse
        self.eb = electron_bunch

        # Параметры по умолчанию
        self.ray_x_angle_range = 5.0e-4
        self.ray_y_angle_range = 5.0e-4
        self.min_energy = 4.005e-15
        self.max_energy = 5.607e-15
        self.np_geometric_factor = 50000
        self.np_emittance = 30000
        self.shift_factor = 1.0
        self.precision = 1.0e-4
        self.total_flux = 0.0
        self.geometric_factor = 1.0
        self.e_spread = False
        self.is_monte_carlo = True
        self.is_compton = True
        self.partial_flux = 0.0
        self.thread_number = max(1, threading.active_count() or 4)
        self.monte_carlo_counter = 0
        self.ksi = None

        self.calculate_total_flux()
        self.calculate_geometric_factor()

    def copy(self) -> 'ThompsonSource':
        new_copy = ThompsonSource(self.lp.copy(), self.eb.copy())
        new_copy.ray_x_angle_range = self.ray_x_angle_range
        new_copy.ray_y_angle_range = self.ray_y_angle_range
        new_copy.min_energy = self.min_energy
        new_copy.max_energy = self.max_energy
        new_copy.np_geometric_factor = self.np_geometric_factor
        new_copy.np_emittance = self.np_emittance
        new_copy.shift_factor = self.shift_factor
        new_copy.precision = self.precision
        new_copy.geometric_factor = self.geometric_factor
        new_copy.e_spread = self.e_spread
        new_copy.is_monte_carlo = self.is_monte_carlo
        new_copy.is_compton = self.is_compton
        new_copy.ksi = self.ksi.copy() if self.ksi else None
        new_copy.thread_number = self.thread_number
        return new_copy

    # Основные методы расчета
    def calculate_total_flux(self) -> None:
        """Расчет полного потока излучения"""
        denom = math.pi * math.sqrt(
            (self.lp.get_width(0) + self.eb.get_x_width(0)) *
            (self.lp.get_width(0) + self.eb.get_y_width(0))
        )
        self.total_flux = (self.SIGMA_T * self.eb._number *
                           self.lp.photon_number * self.lp.fq / denom)

    def calculate_angle_total_flux(self, max_angle: float) -> float:
        """Поток в заданном угле"""
        gamma = self.eb.gamma
        gamma2 = gamma * gamma
        v = math.sqrt(1.0 - 1.0 / gamma2)
        v2 = v * v
        cs = math.cos(max_angle)

        term1 = (1.0 - cs) / (1.0 - v * cs) / (1.0 - v)
        term2 = (0.833333 + 0.166667 / v2 -
                 0.166667 / gamma2 / v2 * (1.0 - v2 * cs) / (1.0 - v) / (1.0 - v * cs))
        term3 = 0.166667 / gamma2 / v2 * (1.0 - cs * cs) / (1.0 - v * cs) ** 3

        return 0.75 / gamma2 * (term1 * term2 + term3) * self.total_flux * self.geometric_factor

    def calculate_geometric_factor(self) -> None:
        """Расчет геометрического фактора методом Монте-Карло"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_number) as executor:
            futures = []
            sum_result = 0.0

            # Параметры области интегрирования
            mult = 3
            wdx = (mult * self.eb.get_x_width(0) * self.lp.get_width(0) /
                   math.sqrt(self.eb.get_x_width(0) + self.lp.get_width(0)))
            wdy = (mult * self.eb.get_y_width(0) * self.lp.get_width(0) /
                   math.sqrt(self.eb.get_y_width(0) + self.lp.get_width(0)))
            len_ = (mult * self.eb.length * self.lp.length /
                    math.sqrt(self.eb.length ** 2 + self.lp.length ** 2))

            it_number = max(1, self.np_geometric_factor // self.thread_number)

            # Запуск расчетов в потоках
            for _ in range(self.thread_number):
                futures.append(executor.submit(
                    self._calculate_geometric_factor_worker,
                    wdx, wdy, len_, it_number
                ))

            # Сбор результатов
            for future in concurrent.futures.as_completed(futures):
                sum_result += future.result()

            self.geometric_factor = (8.0 * wdx * wdy * len_ *
                                     sum_result / it_number / self.thread_number)

    def _calculate_geometric_factor_worker(self, wdx: float, wdy: float, len_: float, it_number: int) -> float:
        """Воркер для расчета геометрического фактора"""
        psum = 0.0
        for _ in range(it_number):


            x = self.eb.shift.x / 2.0 + wdx * (2.0 * random.random() - 1.0)
            y = self.eb.shift.y / 2.0 + wdy * (2.0 * random.random() - 1.0)
            z = self.eb.shift.z / 2.0 + len_ * (2.0 * random.random() - 1.0)

            psum += self.volume_flux(Vector(x, y, z))

        return psum

    def volume_flux(self, r: Vector) -> float:
        """Объемный поток в точке r"""
        len_total = math.sqrt(self.lp.length ** 2 + self.eb.length ** 2)
        sn = self.lp.direction.y
        cs = self.lp.direction.z

        x0, y0, z0 = self.eb.shift.x, self.eb.shift.y, self.eb.shift.z
        x, y, z = r.x, r.y, r.z

        # Преобразование координат
        x1 = x
        y1 = -sn * z + cs * y
        z1 = cs * z + sn * y

        # Расчет экспоненты
        K = ((z + z1 - z0 - self.lp.delay) / len_total) ** 2
        K += (x - x0) ** 2 / self.eb.get_x_width(z - z0)
        K += (y - y0) ** 2 / self.eb.get_y_width(z - z0)
        K += (x1 ** 2 + y1 ** 2) / self.lp.get_width(z1)

        # Расчет коэффициента
        numerator = 2.0 / math.pi ** 1.5
        numerator *= math.sqrt(
            (self.lp.get_width(0) + self.eb.get_x_width(0)) *
            (self.lp.get_width(0) + self.eb.get_y_width(0))
        )
        numerator /= len_total * self.lp.get_width(z1)
        numerator /= self.eb.get_x_width(z - z0)
        numerator /= self.eb.get_y_width(z - z0)

        u = numerator * math.exp(-K)
        return u if not math.isnan(u) else 0.0

    # Методы для расчета направленных потоков
    def direction_flux(self, n: Vector, v: Vector) -> float:
        """Поток в заданном направлении"""
        th = (1.0 - n.inner_product(v)) * 2.0
        gamma2 = self.eb.gamma * self.eb.gamma

        if not self.is_compton:
            return (self.total_flux * 3.0 / 2.0 / math.pi * gamma2 *
                    (1.0 + (th * gamma2) ** 2) / (1.0 + gamma2 * th) ** 4 *
                    self.geometric_factor)
        else:
            ac = self.lp.photon_energy / (self.MC2 * 1.602e-19)  # Приведение к безразмерным единицам
            return (self.total_flux * 3.0 / 2.0 / math.pi * gamma2 *
                    (1.0 + (th * gamma2) ** 2) / (1.0 + gamma2 * th) ** 2 /
                    (1.0 + gamma2 * th + 4.0 * ac * self.eb.gamma) ** 2 *
                    self.geometric_factor)

    def direction_energy(self, n: Vector, v: Vector) -> float:
        """Энергия в заданном направлении"""
        mv = math.sqrt(1.0 - 1.0 / self.eb.gamma ** 2)
        cs = n.inner_product(v)

        if not self.is_compton:
            return (1.0 + mv) * self.lp.photon_energy / (1.0 - cs * mv)
        else:
            ac = self.lp.photon_energy / (self.MC2 * 1.602e-19)
            return ((1.0 + mv) * self.lp.photon_energy /
                    (1.0 - cs * mv + ac * self.eb.gamma * (1.0 + cs)))

    # Методы для расчета поляризации
    def get_polarization(self, ksi_vector: List[float]) -> List[float]:
        """Расчет параметров поляризации"""
        pol = [0.0] * 4
        p = math.sqrt(sum(k ** 2 for k in ksi_vector))
        p = min(p, 1.0)

        if ksi_vector[0] == -p:
            pol[0] = math.sqrt((1.0 - ksi_vector[0]) / 2.0)
            pol[1] = math.sqrt((1.0 + ksi_vector[0]) / 2.0)
            pol[2] = random.random() * 2.0 * math.pi
            pol[3] = random.random() * 2.0 * math.pi
        else:
            k1 = math.sqrt(1.0 - p)
            k2 = math.sqrt(1.0 + p)
            coef = math.sqrt((p + ksi_vector[0]) / p) / 2.0

            # Использование комплексных чисел для расчета фазы
            ksi_complex = complex(ksi_vector[1], ksi_vector[2])
            phase1 = cmath.exp(1j * random.random() * 2.0 * math.pi)
            phase2 = cmath.exp(1j * random.random() * 2.0 * math.pi)

            e1 = phase1 * k1 + phase2 * ksi_complex * k2 / (p + ksi_vector[0])
            e1 *= coef

            e2 = phase2 * k2 - phase1 * ksi_complex.conjugate() * k1 / (p + ksi_vector[0])
            e2 *= coef

            pol[0] = abs(e1)
            pol[1] = abs(e2)
            pol[2] = cmath.phase(e1)
            pol[3] = cmath.phase(e2)

        return pol

    # Методы Монте-Карло
    def get_ray(self) -> List[float]:
        """Генерация одного луча методом Монте-Карло"""
        ray = [0.0] * self.NUMBER_OF_COLUMNS
        prob0 = 0.0
        MULT = 2.0
        sum_prob = 0.0
        lcn = 0

        # Направление по умолчанию
        n = Vector(0, 0, 1)
        r0 = Vector(0, 0, 0)

        # Максимальная энергия
        E_max = self.direction_energy(n, n)

        # Расчет фактора нормализации
        factor = (32.0 * MULT ** 3 *
                  max(self.eb.get_x_width(0), self.lp.get_width(0)) *
                  max(self.eb.get_y_width(0), self.lp.get_width(0)) *
                  max(self.eb.length, self.lp.length) *
                  self.ray_x_angle_range * self.ray_y_angle_range *
                  (self.max_energy - self.min_energy))

        # Основной цикл Монте-Карло
        while True:
            if threading.current_thread().is_interrupted():
                raise KeyboardInterrupt("Thread interrupted")

            # Генерация случайной точки
            ray[0] = MULT * (2.0 * random.random() - 1.0) * max(self.eb.get_x_width(0), self.lp.get_width(0))
            ray[2] = MULT * (2.0 * random.random() - 1.0) * max(self.eb.get_y_width(0), self.lp.get_width(0))
            ray[1] = MULT * (2.0 * random.random() - 1.0) * max(self.eb.length, self.lp.length)

            # Генерация случайного направления
            ray[3] = self.ray_x_angle_range * (2.0 * random.random() - 1.0)
            ray[5] = self.ray_y_angle_range * (2.0 * random.random() - 1.0)

            n = Vector(ray[3], ray[5], 1.0).normalize()
            ray[3], ray[5], ray[4] = n.x, n.y, n.z

            # Генерация случайной энергии
            ray[10] = random.random() * (self.max_energy - self.min_energy) + self.min_energy

            # Расчет вероятности принятия
            if self.e_spread:
                thetax = MULT * self.eb.get_x_spread() * (2.0 * random.random() - 1.0)
                thetay = MULT * self.eb.get_y_spread() * (2.0 * random.random() - 1.0)
                v = Vector(thetax, thetay, math.sqrt(1.0 - thetax ** 2 - thetay ** 2))

                if self.ksi is None:
                    pol_param = self.direction_frequency_volume_polarization_no_spread(r0, n, v, ray[10])
                else:
                    flux = self.direction_frequency_volume_flux_no_spread(r0, n, v, ray[10])
                    pol_param = [flux, self.ksi[0], self.ksi[1], self.ksi[2]]

                prob = pol_param[0] * self.eb.angle_distribution(thetax, thetay) / ray[10]
            else:
                v = Vector(0, 0, 1)

                if self.ksi is None:
                    pol_param = self.direction_frequency_volume_polarization_no_spread(r0, n, v, ray[10])
                else:
                    flux = self.direction_frequency_volume_flux_no_spread(r0, n, v, ray[10])
                    pol_param = [flux, self.ksi[0], self.ksi[1], self.ksi[2]]

                prob = pol_param[0] / ray[10]

            # Критерий принятия
            if not math.isnan(prob) and (lcn == 0 or prob / prob0 > random.random()):
                break

            if not math.isnan(prob):
                sum_prob += prob

            lcn += 1

        # Расчет поляризации
        n = Vector(ray[3], ray[4], ray[5])
        n0 = Vector(0, 1, 0)
        T = self.get_transform(n, n0)

        if self.ksi is not None:
            pol = self.get_polarization(self.ksi)
        else:
            ksi_norm = [p / pol_param[0] for p in pol_param[1:4]]
            pol = self.get_polarization(ksi_norm)

        # Заполнение параметров луча
        As = T @ Vector(1, 0, 0) * pol[0]
        ray[6], ray[7], ray[8] = As.x, As.y, As.z

        As = T @ Vector(0, 0, 1) * pol[1]
        ray[15], ray[16], ray[17] = As.x, As.y, As.z

        ray[9] = 1.0
        ray[13], ray[14] = pol[2], pol[3]

        # Обновление счетчиков
        self.partial_flux += sum_prob * factor
        self.monte_carlo_counter += lcn

        return ray

    def get_transform(self, n: Vector, n0: Vector) -> np.ndarray:
        """Матрица преобразования для поляризации"""
        I = np.eye(3)
        inner = n.inner_product(n0)

        # Матрица D
        D = (np.outer(n.to_array(), n0.to_array()) +
             np.outer(n0.to_array(), n.to_array())) * inner
        D -= (np.outer(n.to_array(), n.to_array()) +
              np.outer(n0.to_array(), n0.to_array()))
        D /= inner ** 2 - 1.0

        # Матрица A
        A = (np.outer(n.to_array(), n0.to_array()) -
             np.outer(n0.to_array(), n.to_array())) + I * inner

        return (I - D) * (1.0 - inner) + A




    def direction_frequency_flux_no_spread(self, n: Vector, v: Vector, e: float) -> float:
        """Расчет спектральной плотности потока без учета разброса"""
        th = (1.0 - n.inner_product(v)) * 2.0

        if not self.is_compton:
            # Томсоновское рассеяние
            K = (math.sqrt(
                e / self.lp.photon_energy / (1.0 - e * th / self.lp.photon_energy / 4.0)) - 2.0 * self.eb.gamma) ** 2
            K /= 4.0 * (self.eb.gamma * self.eb.delgamma) ** 2

            term1 = math.sqrt(e / self.lp.photon_energy)
            term2 = (1.0 - e * th / self.lp.photon_energy / 2.0) ** 2 + 1.0
            term3 = math.sqrt(1.0 - e * th / self.lp.photon_energy / 4.0)

            res = (self.total_flux * e * 3.0 / 64.0 / math.pi / math.sqrt(math.pi) /
                   self.eb.delgamma / self.eb.gamma / self.lp.photon_energy *
                   term1 * term2 / term3 * math.exp(-K))
        else:
            # Комптоновское рассеяние
            ac = self.lp.photon_energy / (self.MC2 * 1.602e-19)  # α_c = ħω/mc^2
            koef = 4.0 * self.lp.photon_energy / e - th
            gamma = (2.0 * ac + math.sqrt(4.0 * ac ** 2 + koef)) / koef
            gamma2 = gamma ** 2

            term1 = self.lp.photon_energy / e ** 2
            term2 = self.eb.gamma ** 5 / (1.0 + gamma2 * th + 4.0 * ac * self.eb.gamma) ** 2
            term3 = (1.0 + ((1.0 - gamma2 * th) / (1.0 + gamma2 * th)) ** 2) / (1.0 + 2.0 * gamma * ac)
            K = ((gamma - self.eb.gamma) / (self.eb.delgamma * self.eb.gamma)) ** 2

            res = (self.total_flux * e * 1.5 / math.pi ** 1.5 / self.eb.delgamma /
                   self.eb.gamma * term1 * term2 * term3 * math.exp(-K))

        return res if not math.isnan(res) else 0.0

    def direction_frequency_polarization_no_spread(self, n: Vector, v: Vector, e: float) -> List[float]:
        """Расчет параметров Стокса без учета разброса"""
        stocks = [0.0] * 4
        th = (1.0 - n.inner_product(v)) * 2.0

        if not self.is_compton:
            # Томсоновское рассеяние
            K = (math.sqrt(
                e / self.lp.photon_energy / (1.0 - e * th / self.lp.photon_energy / 4.0)) - 2.0 * self.eb.gamma) ** 2
            K /= 4.0 * (self.eb.gamma * self.eb.delgamma) ** 2

            m11 = (self.total_flux * e * 3.0 / 32.0 / math.pi / math.sqrt(math.pi) /
                   self.eb.delgamma / self.eb.gamma / self.lp.photon_energy *
                   math.sqrt(e / self.lp.photon_energy) /
                   math.sqrt(1.0 - e * th / self.lp.photon_energy / 4.0) *
                   math.exp(-K))

            mlt = 1.0 - e * th / self.lp.photon_energy / 2.0
        else:
            # Комптоновское рассеяние
            ac = self.lp.photon_energy / (self.MC2 * 1.602e-19)
            koef = 4.0 * self.lp.photon_energy / e - th
            gamma = (2.0 * ac + math.sqrt(4.0 * ac ** 2 + koef)) / koef
            gamma2 = gamma ** 2

            m11 = (self.total_flux * e * 3.0 / math.pi ** 1.5 / self.eb.delgamma /
                   self.eb.gamma * self.lp.photon_energy / e ** 2 *
                   self.eb.gamma ** 5 / (1.0 + gamma2 * th + 4.0 * ac * self.eb.gamma) ** 2 /
                   (1.0 + 2.0 * gamma * ac) * math.exp(-((gamma - self.eb.gamma) /
                                                         (self.eb.delgamma * self.eb.gamma)) ** 2))

            mlt = (1.0 - gamma2 * th) / (1.0 + gamma2 * th)

        m12 = m11 * mlt
        m22 = m12 * mlt

        # Преобразование в систему координат наблюдателя
        vn = v.inner_product(n)
        norm = math.sqrt((1.0 - vn ** 2) * (1.0 - n.x ** 2))

        if norm != 0.0:
            cs = (v.x - n.x * vn) / norm
            sn = (n.y * v.z - n.z * v.y) / norm
        else:
            cs = 1.0
            sn = 0.0

        cs2 = 2.0 * cs ** 2 - 1.0
        sn2 = 2.0 * sn * cs

        # Параметры Стокса
        stocks[0] = (m11 + m22 - (cs2 * self.lp.polarization[2] + sn2 * self.lp.polarization[0]) *
                     (m11 - m22)) / 2.0

        stocks[3] = (cs2 * (m22 - m11) + self.lp.polarization[2] * (cs2 ** 2 * (m11 + m22) +
                                                                    2.0 * sn2 ** 2 * m12) + self.lp.polarization[
                         0] * cs2 * sn2 *
                     (m11 + m22 - 2.0 * m12)) / 2.0

        stocks[1] = (sn2 * (m22 - m11) + self.lp.polarization[0] * (sn2 ** 2 * (m11 + m22) +
                                                                    2.0 * cs2 ** 2 * m12) + self.lp.polarization[
                         2] * cs2 * sn2 *
                     (m11 + m22 - 2.0 * m12)) / 2.0

        stocks[2] = self.lp.polarization[1] * m12

        # Проверка на корректность
        for i in range(4):
            if math.isnan(stocks[i]) or math.isnan(stocks[0]) or stocks[0] <= 0.0:
                stocks[i] = 0.0

        return stocks

    def direction_frequency_flux_spread_integral(self, n: Vector, v0: Vector, e: float) -> float:
        """Расчет с учетом разброса через интегрирование"""

        def integrand(y: float, x: float) -> float:
            v = Vector(x, y, math.sqrt(1.0 - x ** 2 - y ** 2))
            dv = v - v0
            flux = self.direction_frequency_flux_no_spread(n, v, e)
            angle_dist = self.eb.angle_y_distribution(dv.y)
            return flux * angle_dist + self.shift_factor * self.SHIFT

        try:
            res, _ = quad(
                lambda x: quad(
                    lambda y: integrand(y, x),
                    -3.0 * self.eb.get_y_spread(),
                    3.0 * self.eb.get_y_spread()
                )[0] * self.eb.angle_x_distribution(x - v0.x),
                -3.0 * self.eb.get_x_spread(),
                3.0 * self.eb.get_x_spread(),
                limit=self.MAXIMAL_NUMBER_OF_EVALUATIONS
            )

            res -= self.shift_factor * self.SHIFT * 36.0 * self.eb.get_x_spread() * self.eb.get_y_spread()
            return res if not math.isnan(res) else 0.0
        except:
            return 0.0

    def direction_frequency_flux_spread_monte_carlo(self, n: Vector, v0: Vector, e: float) -> float:
        """Расчет с учетом разброса методом Монте-Карло"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_number) as executor:
            futures = []
            sum_result = 0.0
            it_number = max(1, self.np_emittance // self.thread_number)

            for _ in range(self.thread_number):
                futures.append(executor.submit(
                    self._monte_carlo_flux_worker,
                    n, v0, e, it_number
                ))

            for future in concurrent.futures.as_completed(futures):
                sum_result += future.result()

            res = 36.0 * self.eb.get_x_spread() * self.eb.get_y_spread() * sum_result / it_number / self.thread_number
            return res if not math.isnan(res) else 0.0

    def _monte_carlo_flux_worker(self, n: Vector, v0: Vector, e: float, it_number: int) -> float:
        """Воркер для метода Монте-Карло"""
        psum = 0.0
        for _ in range(it_number):
            if threading.current_thread().is_interrupted():
                return psum

            rx = (2.0 * random.random() - 1.0) * 3.0 * self.eb.get_x_spread()
            ry = (2.0 * random.random() - 1.0) * 3.0 * self.eb.get_y_spread()
            v = Vector(rx, ry, math.sqrt(1.0 - rx ** 2 - ry ** 2))
            dv = v - v0

            flux = self.direction_frequency_flux_no_spread(n, v, e)
            angle_dist = self.eb.angle_distribution(dv.x, dv.y)
            psum += flux * angle_dist if not math.isnan(flux * angle_dist) else 0.0

        return psum

    def direction_frequency_volume_flux_no_spread(self, r: Vector, n: Vector, v: Vector, e: float) -> float:
        """Объемная спектральная плотность без разброса"""
        return self.direction_frequency_flux_no_spread(n, v, e) * self.volume_flux(r)

    def direction_frequency_volume_polarization_no_spread(self, r: Vector, n: Vector, v: Vector, e: float) -> List[
        float]:
        """Объемные параметры Стокса без разброса"""
        stocks = self.direction_frequency_polarization_no_spread(n, v, e)
        v_flux = self.volume_flux(r)
        return [s * v_flux for s in stocks]

    def set_ray_ranges(self, xangle: float, yangle: float, min_en: float, max_en: float) -> None:
        """Установка диапазонов для генерации лучей"""
        self.ray_x_angle_range = xangle
        self.ray_y_angle_range = yangle
        self.min_energy = min_en
        self.max_energy = max_en

    def set_polarization(self, ksi: List[float]) -> None:
        """Установка вектора поляризации"""
        self.ksi = ksi.copy()

    def get_approx_geometric_factor(self) -> float:
        """Приближенный расчет геометрического фактора"""
        cs2 = (1.0 + self.lp.direction.inner_product(Vector(0, 0, 1))) / 2.0
        sn2 = (1.0 - self.lp.direction.inner_product(Vector(0, 0, 1))) / 2.0
        w2 = self.lp.get_width2(0) + self.eb.get_width2(0)
        l2 = self.lp.length ** 2 + self.eb.length ** 2
        return math.sqrt(w2 / (l2 * sn2 + w2 * cs2) / cs2)

    def __repr__(self) -> str:
        return (f"ThompsonSource(total_flux={self.total_flux:.3e}, "
                f"geometric_factor={self.geometric_factor:.3f}, "
                f"threads={self.thread_number})")