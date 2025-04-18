from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np  # Для удобной работы с 2D-массивами


class ColorChartParam(ABC):
    def __init__(
            self,
            u: Optional[List[List[float]]] = None,
            um: float = 0.0
    ) -> None:
        """
        Конструктор.
        :param u: 2D-массив данных (если None, будет создан в setup).
        :param um: Максимальное значение.
        """
        self.__udata: Optional[List[List[float]]] = u
        self.__umax: float = um

        # Параметры сетки (инициализируются в setup)
        self.__xoffset: float = 0.0
        self.__yoffset: float = 0.0
        self.__xstep: float = 0.0
        self.__ystep: float = 0.0
        self.__xsize: int = 0
        self.__ysize: int = 0

    def setup(
            self,
            xsize: int,
            ysize: int,
            xstep: float,
            ystep: float,
            xoffset: float,
            yoffset: float
    ) -> None:
        """
        Инициализирует udata на основе func(x, y).
        :raises KeyboardInterrupt: Если поток прерван (аналог InterruptedException).
        """
        self.__udata = [[0.0 for _ in range(ysize)] for _ in range(xsize)]
        self.__xoffset = xoffset
        self.__yoffset = yoffset
        self.__xstep = xstep
        self.__ystep = ystep
        self.__xsize = xsize
        self.__ysize = ysize

        try:
            for j in range(xsize):
                for p in range(ysize):
                    # Проверка прерывания (в Python это KeyboardInterrupt)
                    # Обычно обрабатывается через сигналы, но для примера:
                    # if threading.current_thread().is_interrupted():
                    #     raise KeyboardInterrupt()

                    x = xoffset + xstep * (j - xsize / 2)
                    y = yoffset + ystep * (p - ysize / 2)
                    self.__udata[j][p] = self.func(x, y)

            self.__umax = self.__udata[xsize // 2][ysize // 2]

        except KeyboardInterrupt:
            print("Прерывание: остановка вычислений.")
            raise

    @abstractmethod
    def func(self, x: float, y: float) -> float:
        """Абстрактный метод для вычисления значения в точке (x, y)."""
        pass

    # Геттеры (в Python используем @property)
    @property
    def udata(self) -> Optional[List[List[float]]]:
        return self.__udata

    @property
    def umax(self) -> float:
        return self.__umax

    @property
    def xoffset(self) -> float:
        return self.__xoffset

    @property
    def yoffset(self) -> float:
        return self.__yoffset

    @property
    def xstep(self) -> float:
        return self.__xstep

    @property
    def ystep(self) -> float:
        return self.__ystep

    @property
    def xsize(self) -> int:
        return self.__xsize

    @property
    def ysize(self) -> int:
        return self.__ysize