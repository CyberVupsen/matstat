import numpy as np
# import matplotlib.pyplot as plt
from matstat_facilities import *
from densities_distributions import *
import math


n = 25
lambda_param = 1

sample = np.random.exponential(scale=1/lambda_param, size=n)

print("Выборка:")
print(sample)

def p_ar_mean(x):
    value = (1/(((1/n)*2*np.pi)**0.5)) * np.exp(-((x - 1)**2)/(2*(1/n)))
    # print("value =", value)
    return value

def F_median_teor(x):
    if x >= 0:
        # print("F =", (1 - np.exp(-x)))
        return (1 - np.exp(-x))
    else:
        # print("F =", 0)
        return 0

def p_median_teor(x):
    k = int((n - 1) / 2)
    value = (p(x) * math.factorial(n)/(math.factorial(k) * math.factorial(n - k - 1))
             * (F_median_teor(x)**k) * (1 - F_median_teor(x))**(n - k - 1))
    # print("p_median_teor =", value)
    return value



# ТЕСТЫ
moda = get_moda(sample)
print("Мода:")
print(moda)

median = get_median(sample)
print("Медиана =", median)

scope = get_scope(sample)
print("Размах =", scope)

kf_asim = get_kf_asim(sample) # значение отличается в десятых от сайта!
print("Коэффициент асимметрии =", kf_asim)

show_emp_distr_func(sample)
show_hist(sample, "Гистограмма выборки объёма " + str(n))
show_boxplot(sample, "Выборка объёма " + str(n))

# сравнение бутстрапа средниих арифметических с ЦПТ (простейшей)
ar_means = gener_bootstr_mass(sample, 1000, get_ar_mean)
# print(ar_means)
show_bootstr_compar_mean(sample, ar_means, p_ar_mean, 50)

# поиск оценки плотности распределения коэффициента ассиметрии бутстрапом (график)
kf_asims = gener_bootstr_mass(sample, 1000, get_kf_asim)
# поиск вероятности того, что коэффициент асимметрии будет меньше 1
chance = emp_distr_func(kf_asims, 1)
print("P(kf_asim < 1) =", chance)
# show_emp_distr_func(kf_asims)
show_hist(kf_asims, "Оценка плотности распред. коэф. асимметрии")

# сравнение плотности распределения медианы выборки с бутстраповской оценкой этой плотности
medians = gener_bootstr_mass(sample, 1000, get_median)
# print(medians)
show_bootstr_compar_median(sample, medians, p_median_teor, 50)
