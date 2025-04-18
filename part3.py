import colorsys
from typing import Optional, Union
import numpy as np
from abc import ABC, abstractmethod


class PaintScale(ABC):
    """Абстрактный базовый класс для цветовых шкал (аналог org.jfree.chart.renderer.PaintScale)"""

    @abstractmethod
    def get_lower_bound(self) -> float:
        pass

    @abstractmethod
    def get_upper_bound(self) -> float:
        pass

    @abstractmethod
    def get_paint(self, value: float) -> tuple:
        pass


class JetPaintScale(PaintScale):
    """Реализация цветовой шкалы 'Jet' (аналог Matlab's jet colormap)"""

    def __init__(self, lower_bound: float = 0.0, upper_bound: float = 1.0):
        """
        Инициализация цветовой шкалы

        :param lower_bound: Нижняя граница шкалы
        :param upper_bound: Верхняя граница шкалы
        :raises ValueError: Если lower_bound >= upper_bound
        """
        if lower_bound >= upper_bound:
            raise ValueError("Requires lower_bound < upper_bound")

        self._lower_bound = float(lower_bound)
        self._upper_bound = float(upper_bound)

    @property
    def lower_bound(self) -> float:
        """Получить нижнюю границу шкалы"""
        return self._lower_bound

    @property
    def upper_bound(self) -> float:
        """Получить верхнюю границу шкалы"""
        return self._upper_bound

    def get_paint(self, value: float) -> tuple:
        """
        Получить цвет для заданного значения

        :param value: Входное значение в диапазоне [lower_bound, upper_bound]
        :return: Цвет в формате RGB (tuple из 3 float значений 0-1)
        """
        # Нормализуем значение в диапазон [0, 1]
        normalized = np.clip(value, self._lower_bound, self._upper_bound)
        normalized = (normalized - self._lower_bound) / (self._upper_bound - self._lower_bound)

        # Преобразуем в цветовое пространство HSV и затем в RGB
        # Используем инвертированное значение для соответствия Java-версии
        h = 1.0 - normalized
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 0.8)

        return (r, g, b)

    def __eq__(self, other: object) -> bool:
        """Проверка на равенство с другим объектом JetPaintScale"""
        if not isinstance(other, JetPaintScale):
            return False
        return (self._lower_bound == other._lower_bound and
                self._upper_bound == other._upper_bound)

    def __repr__(self) -> str:
        """Строковое представление объекта"""
        return f"JetPaintScale(lower_bound={self._lower_bound}, upper_bound={self._upper_bound})"

    def copy(self) -> 'JetPaintScale':
        """Создание копии объекта"""
        return JetPaintScale(self._lower_bound, self._upper_bound)


# Пример использования
if __name__ == "__main__":
    try:
        # Создание цветовой шкалы
        scale = JetPaintScale(0.0, 100.0)

        # Получение цветов для разных значений
        print(scale.get_paint(0.0))  # (0.8, 0.0, 0.0) - красный
        print(scale.get_paint(25.0))  # Оранжевый
        print(scale.get_paint(50.0))  # Желтый
        print(scale.get_paint(75.0))  # Голубой
        print(scale.get_paint(100.0))  # (0.0, 0.0, 0.8) - синий

        # Проверка равенства
        scale2 = JetPaintScale(0.0, 100.0)
        print(scale == scale2)  # True

        # Проверка ошибки при создании
        try:
            bad_scale = JetPaintScale(10.0, 5.0)
        except ValueError as e:
            print(f"Ошибка: {e}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")