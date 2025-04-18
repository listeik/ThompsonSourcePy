import numpy as np
from typing import List, Callable, Optional, Union
import threading


class LinearChartParam:
    """Класс для линейных параметров графика с поддержкой данных и функций"""

    def __init__(self):
        """Инициализация параметров"""
        self._size: int = 0
        self._step: float = 0.0
        self._offset: float = 0.0
        self._data: Optional[np.ndarray] = None
        self._umax: float = 0.0
        self._umin: float = 0.0
        self._func: Optional[List[Callable[[float], float]]] = None

    @property
    def size(self) -> int:
        """Количество точек данных"""
        return self._size

    @property
    def step(self) -> float:
        """Шаг между точками"""
        return self._step

    @property
    def offset(self) -> float:
        """Смещение по оси X"""
        return self._offset

    @property
    def data(self) -> Optional[np.ndarray]:
        """Двумерный массив данных (строки - функции, столбцы - точки)"""
        return self._data

    @property
    def umax(self) -> float:
        """Максимальное значение данных"""
        return self._umax

    @property
    def umin(self) -> float:
        """Минимальное значение данных"""
        return self._umin

    def setup_from_data(
            self,
            data: np.ndarray,
            index: int,
            row: bool,
            size: int,
            step: float,
            offset: float
    ) -> None:
        """
        Инициализация из массива данных

        :param data: Исходный массив данных
        :param index: Индекс строки/столбца для выборки
        :param row: True - выборка строки, False - выборка столбца
        :param size: Количество точек
        :param step: Шаг между точками
        :param offset: Смещение
        :raises KeyboardInterrupt: Если поток был прерван
        """
        self._size = size
        self._step = step
        self._offset = offset
        self._data = np.zeros((1, size))
        self._func = None

        try:
            for i in range(size):
                # Проверка прерывания потока
                if threading.current_thread().is_interrupted():
                    raise KeyboardInterrupt("Thread interrupted")

                self._data[0, i] = data[i, index] if row else data[index, i]

            self._set_extrema()

        except KeyboardInterrupt:
            self._data = None
            raise

    def setup_from_functions(
            self,
            funcs: List[Callable[[float], float]],
            size: int,
            step: float,
            offset: float
    ) -> None:
        """
        Инициализация из списка функций

        :param funcs: Список функций f(x) -> y
        :param size: Количество точек
        :param step: Шаг между точками
        :param offset: Смещение
        :raises KeyboardInterrupt: Если поток был прерван
        """
        self._size = size
        self._step = step
        self._offset = offset
        self._func = funcs
        self._data = np.zeros((len(funcs), size))

        try:
            for i in range(size):
                xp = step * i + offset

                # Проверка прерывания потока
                if threading.current_thread().is_interrupted():
                    raise KeyboardInterrupt("Thread interrupted")

                for j, func in enumerate(funcs):
                    self._data[j, i] = func(xp)

            self._set_extrema()

        except KeyboardInterrupt:
            self._data = None
            raise

    def _set_extrema(self) -> None:
        """Вычисление минимального и максимального значений"""
        if self._data is None:
            self._umax = 0.0
            self._umin = 0.0
            return

        sz = len(self._func) if self._func is not None else 1

        umaxt = np.zeros(sz)
        umint = np.zeros(sz)

        for k in range(sz):
            umaxt[k] = np.max(self._data[k])
            umint[k] = np.min(self._data[k])

        self._umax = np.max(umaxt)
        self._umin = np.min(umint)

    def __repr__(self) -> str:
        return (f"LinearChartParam(size={self.size}, step={self.step}, "
                f"offset={self.offset}, umin={self.umin:.3f}, umax={self.umax:.3f})")


# Пример использования
if __name__ == "__main__":
    try:
        # Пример 1: Инициализация из данных
        data_2d = np.random.rand(10, 10)  # Тестовые данные 10x10
        param1 = LinearChartParam()
        param1.setup_from_data(data_2d, index=2, row=True, size=10, step=0.1, offset=0.0)
        print(f"Пример 1: {param1}")
        print(f"Данные:\n{param1.data}")


        # Пример 2: Инициализация из функций
        def func1(x):
            return x ** 2


        def func2(x):
            return np.sin(x)


        param2 = LinearChartParam()
        param2.setup_from_functions([func1, func2], size=100, step=0.1, offset=-5.0)
        print(f"\nПример 2: {param2}")
        print(f"Максимальное значение: {param2.umax:.3f}")

    except Exception as e:
        print(f"Ошибка: {e}")