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
    a_1 = list(map(float, a_1.split()))
    a_2 = list(map(float, a_2.split()))

    def f(x):
        return a_1[0] * x ** 2 + a_1[1] * x + a_1[2]

    def p(x):
        return a_2[0] * x ** 2 + a_2[1] * x + a_2[2]

    res_f = minimize_scalar(f)
    res_p = minimize_scalar(p)

    solutions = []
    to_inf = 0
    for x in range(-100, 101):
        if abs(f(x) - p(x)) < 1e-5:
            solutions.append((x, f(x)))
        to_inf += 1

    if not solutions:
        return []
    if len(solutions) == to_inf:
        return None
    return solutions




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
