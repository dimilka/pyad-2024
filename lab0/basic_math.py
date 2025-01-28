import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы нельзя перемножить: количество столбцов первой матрицы не равно количеству строк второй.")

    m, n, p = len(matrix_a), len(matrix_b), len(matrix_b[0])
    result = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    # Преобразуем строки в список чисел
    coef1 = list(map(int, a_1.split()))
    coef2 = list(map(int, a_2.split()))

    if coef1 == coef2:
        return None

    def f(x, coef):
        return coef[0] * x**2 + coef[1] * x + coef[2]

    #extreme1 = -coef1[1] / (2 * coef1[0]) if coef1[0] != 0 else None
    #extreme2 = -coef2[1] / (2 * coef2[0]) if coef2[0] != 0 else None

    ans = []
    if coef1[0] == coef2[0]:
        if coef1[1] != coef2[1]:
            x = (coef2[2] - coef1[2]) / (coef1[1] - coef2[1])
            ans.append((x, f(x, coef1)))
    else:
        a = coef1[0] - coef2[0]
        b = coef1[1] - coef2[1]
        c = coef1[2] - coef2[2]
        det = b ** 2 - 4 * a * c
        if det >= 0:
            x1 = (-b + (det) ** 0.5) / (2 * a)
            x2 = (-b - (det) ** 0.5) / (2 * a)
            ans.append((x1, f(x1, coef1)))
            if x1 != x2:
                ans.append((x2, f(x2, coef1)))

    return ans





def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean_x = np.mean(x)
    m2 = np.mean((x - mean_x) ** 2)
    m3 = np.mean((x - mean_x) ** 3)

    if m2 == 0:
        return 0.0

    skewness = m3 / (m2 ** 1.5)
    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    n = len(x)
    mean_x = np.mean(x)
    m2 = np.mean((x - mean_x) ** 2)
    m4 = np.mean((x - mean_x) ** 4)

    if m2 == 0:
        return 0.0

    kurt = m4 / (m2 ** 2) - 3
    return round(kurt, 2)
