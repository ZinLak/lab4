"""
Лабораторная работа №4
С клавиатуры вводится два числа K и N. Квадратная матрица А(N,N),
состоящая из 4-х равных по размерам подматриц, B,C,D,E заполняется
случайным образом целыми числами в интервале [-10,10].
Для отладки использовать не случайное заполнение, а целенаправленное.

Для простоты все индексы в подматрицах относительные. 
По сформированной матрице F (или ее частям) необходимо вывести не менее 3 разных графиков.
Программа должна использовать функции библиотек numpy  и mathplotlib

ВАРИАНТ 26.	Формируется матрица F следующим образом: скопировать в нее А и если в С
количество нулей в нечетных столбцах больше, чем произведение чисел по периметру,
то поменять местами  В и С симметрично, иначе В и Е поменять местами несимметрично.
При этом матрица А не меняется. После чего если определитель матрицы А больше суммы
диагональных элементов матрицы F, то вычисляется выражение: A-1*AT – K * F,
иначе вычисляется выражение (A +G-1-F-1)*K, где G-нижняя треугольная матрица,
полученная из А. Выводятся по мере формирования А, F и все матричные операции
последовательно.
"""

import matplotlib.pyplot as plt
import numpy as np

def fill_matrix(matrix, submatrix):
    half_size = len(matrix) // 2
    for i, key in enumerate(['D', 'E', 'C', 'B']):
        row_start = half_size * (i // 2)
        col_start = half_size * (i % 2)
        matrix[row_start:row_start+half_size, col_start:col_start+half_size] = submatrix[key]

def main():

    N = int(input("Введите размерность матрицы N: "))
    K = int(input("Введите коэффициент K: "))

    submatrix = {'D': [], 'E' :[], 'C': [], 'B': []}

    A = np.zeros((N, N), dtype=np.int64)
    F = np.zeros((N, N), dtype=np.int64)

    for key in submatrix: #Заполнение подматриц
        submatrix[key] = np.random.randint(-10, 11, size=(N//2, N//2))
        print('Подматрица', key)
        print(submatrix[key], '\n')

    fill_matrix(A, submatrix)
    print('Матрица А\n', A)

    m = np.copy(submatrix['B'])
    if 'C' in submatrix:
        countZero = 0 # счетчик нулей по нечетным столбцам
        for i in range(0, len(submatrix['C']), 1):
            for j in range(0, len(submatrix['C']), 2):
                if submatrix['C'][i][j] == 0:
                    countZero += 1

        product_of_number = 1 #Произвеедение по периметру C
        for i in range(0, len(submatrix['C'])):
            if (i != 0 or i != (len(submatrix['C']))) and (i > 0 and i < len(submatrix['C'])-1):
                n = len(submatrix['C'])-1
                for j in range(0, len(submatrix['C']), n):
                    product_of_number = product_of_number * submatrix['C'][i][j]
            else:
                for j in range(0, len(submatrix['C'])):
                    product_of_number = product_of_number * submatrix['C'][i][j]   

        if countZero > product_of_number: #Условие
            for i in range(0, len(submatrix['C'])):
                submatrix['B'][i][0::] = submatrix['C'][i][::-1]
                submatrix['C'][i][0::] = m[i][::-1]
            print('Матрица С \n', submatrix['C'])
            print('Матрица В \n', submatrix['B'])
        else:
            submatrix['B'], submatrix['E'] = submatrix['E'], submatrix['B']
            print('Матрица B \n', submatrix['B'])
            print('Матрица E \n', submatrix['E'])

    fill_matrix(F, submatrix)
    print('\nМатрица F\n', F)

    determinant = int(np.linalg.det(A)); print('\nОпределитель матрицы A:', determinant)
    sumDiagF = 0
    for i in range(0, len(F)):
        sumDiagF = sumDiagF+F[i][i]
    if determinant > sumDiagF:
        A_inv = np.linalg.inv(A)
        A_inv = np.round(A_inv, 3); print('\nОбратная\n',A_inv)
        A_transpose = np.transpose(A); print('\nТранспонированная\n',A_transpose)
        result = np.dot(A_inv, A_transpose) - K * F
        result_rounded = np.round(result, 4)
        print('\nРезультат выражения:\n', result_rounded)
    else:
        G = np.tril(A)
        G_inv = np.linalg.pinv(G)
        F_inv = np.linalg.pinv(F)
        result = (A + G_inv - F_inv) * K
        result_rounded = np.round(result, 4)
        print('\nРезультат выражения:\n', result_rounded)

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 4, 1)
    plt.hist(F.flatten(), bins=20, color='skyblue', edgecolor='black')
    plt.title('Гистограмма матрицы F')

    plt.subplot(1, 4, 2)
    labels = ['B', 'C', 'D', 'E']
    sizes = [np.sum(submatrix[key]) for key in labels]
    # Проверяем, что сумма элементов не равна нулю, чтобы избежать деления на ноль
    sizes = [max(0, s) + 0.01 for s in sizes]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Круговая диаграмма подматриц')

    plt.subplot(1, 4, 3)
    plt.plot(np.diag(F), marker='o', color='green')
    plt.title('Линейный график главной диагонали F')

    plt.subplot(1, 4, 4)
    plt.bar(np.arange(len(sizes)), sizes, color='orange')
    plt.xticks(np.arange(len(labels)), labels)
    plt.title('Гистограмма суммы элементов подматриц')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
