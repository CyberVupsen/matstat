import numpy as np
import matplotlib.pyplot as plt
import math
# from random import randint
from densities_distributions import *


#поиск моды - проверил. работает.
def get_moda(sample):
    sample = np.array(sample)
    num_of_replays = np.array([])
    modas = np.array([])

    for i in range(len(sample)):
        replays = 0
        for j in range(len(sample)):
            if sample[i] == sample[j]:
                replays += 1
        num_of_replays = np.append(num_of_replays, (replays - 1))

    max_num_of_replays = np.max(num_of_replays)
    for i in range(len(num_of_replays)):
        if num_of_replays[i] == max_num_of_replays and sample[i] not in modas:
            modas = np.append(modas, sample[i])

    return np.array(modas)

#поиск медианы
def get_median(sample):
    sample = np.array(sample)
    sorted_sample = np.sort(sample)
    # print("Вариационный ряд:")
    # print(sorted_sample)

    if len(sorted_sample) % 2 == 1:
        median = sorted_sample[len(sorted_sample) // 2]
    if len(sorted_sample) % 2 ==0:
        median = (sorted_sample[(len(sorted_sample) // 2) - 1] 
                + sorted_sample[(len(sorted_sample) // 2)]) / 2

    return median

#поиск размаха
def get_scope(sample):
    sample = np.array(sample)
    scope = np.max(sample) - np.min(sample)
    return scope

#поиск математического ожидания (M)
# M = 0
# for i in range(len(sample)):
#     M += (func(x) * x)

#поиск среднего арифметического
def get_ar_mean(sample):
    sum = 0
    for i in sample:
        sum += i
    mean = sum/len(sample)
    return mean

#поиск оценки центрального момента k-того порядка (ev - evaluation)
def get_centr_m_ev(sample, k):
    x_mean = get_ar_mean(sample)
    deviations = [(x - x_mean)**k for x in sample]
    m_ev = get_ar_mean(deviations)
    return m_ev

#поиск коэффициента ассиметрии (kf_asim)
def get_kf_asim(sample):
    kf_asim = get_centr_m_ev(sample, 3) / get_centr_m_ev(sample, 2)**1.5
    return kf_asim

############################################################
#поиск количества элементов, которые меньше по значению, чем x
def less_than(sample, x):
    smaller = 0
    for i in sample:
        if i < x:
            smaller += 1
    return smaller

# эмпирическая функция распределения
def emp_distr_func(sample, x):
    return (less_than(sample, x) / len(sample))

# построение графика эмпирической функции распределения
def show_emp_distr_func(sample):
    sample = np.array(sample)
    sorted_sample = np.sort(sample)
    emp_distr_f_mass = [emp_distr_func(sorted_sample, x) for x in sorted_sample]
    # способ 1:
    # plt.scatter(sample, emp_distr_func)
    # способ 2:
    fig, axes = plt.subplots(1,2)
    axes[0].step(np.concatenate(([0], sorted_sample, [sorted_sample[-1]])),
                 np.concatenate(([0], emp_distr_f_mass, [emp_distr_f_mass[-1]])),
                 where="pre")
    axes[1].ecdf(sorted_sample)
    axes[0].set_xticks(sample)
    axes[0].set_yticks(emp_distr_f_mass)
    axes[0].set_title("Моя эпирич.функц.распр.")
    axes[1].set_title('"Зашитая" эпирич.функц.распр.')
    plt.show()

# построение гистограммы распределения
def show_hist(sample, hist_title):
    fig, axes = plt.subplots(1,2)
    num_of_bins = 1 + int(math.log2(len(sample)))
    # print("num of bins =",num_of_bins)
    # axes[0].hist(sample, num_of_bins, edgecolor='black')
    axes[0].hist(sample, num_of_bins, edgecolor='black', density=True)
    axes[0].set_title(hist_title)
    plt.show()

# построение boxplot
def show_boxplot(sample, boxpl_title):
    fig, axes = plt.subplots(1,2)
    # axes[0].boxplot(sample, vert=False)
    axes[0].boxplot(sample, vert=False, whis=[0, 100], patch_artist=False, showmeans=False)
    # axes[0].set(xticks=[8,12.5,21])
    
    # # Расчет квартилей
    # q1, q2, q3 = np.percentile(sample, [25, 50, 75])

    # # Добавление текста
    # axes[1].text(1.1, q1, f'Q1: {q1:.2f}', va='center')
    # axes[1].text(1.1, q2, f'Q2: {q2:.2f}', va='center')
    # axes[1].text(1.1, q3, f'Q3: {q3:.2f}', va='center')

    # axes[1].set_yticklabels([])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(boxpl_title)
    plt.show()

# определение оценки плотности распреления среднего арифметического элементов выборки бутстрапом
# генерация средних арифметических для бутстрапа
# def gener_ar_means(sample, length):
#     ar_means = []
#     for i in range(length):
#         # numbers = [randint() for i in range(len(sample))]
#         mers_twist = np.random.RandomState()
#         subsample = mers_twist.choice(sample, size=len(sample))
#         print(subsample)
#         ar_means = np.append(ar_means, get_ar_mean(subsample))
#     return ar_means

def gener_bootstr_mass(sample, length, stat_func):
    bootstr_mass = []
    for i in range(length):
        # numbers = [randint() for i in range(len(sample))]
        mers_twist = np.random.RandomState()
        subsample = mers_twist.choice(sample, size=len(sample))
        # print(subsample)
        bootstr_mass = np.append(bootstr_mass, stat_func(subsample))
    return bootstr_mass

def get_unmoved_var(sample):
    s = 0
    x_mean = get_ar_mean(sample)
    for x in sample:
        s += (x - x_mean)**2
    unmoved_var = 1/(len(sample) - 1) * s
    return unmoved_var

# поиск функции плотности распределения среднего арифметического на основе выборки
def get_density_func(sample):
    mat_exp = get_ar_mean(sample) # ещё подумать над этим
    variance = get_ar_mean(sample**2) - get_ar_mean(sample)**2
    n = len(sample)
    def p_ar_mean(x):
        value = (1/((variance * 2 * np.pi)**0.5)) * np.exp(-((x - mat_exp)**2)/(2 * variance))
        # print("value =", value)
        return value
    return p_ar_mean

def get_density_func_unmoved(sample):
    mat_exp = get_ar_mean(sample)
    unmoved_var = get_unmoved_var(sample)
    n = len(sample)
    def p_ar_mean_unmoved(x):
        value = (1/((unmoved_var * 2 * np.pi)**0.5)) * np.exp(-((x - mat_exp)**2)/(2 * unmoved_var))
        # print("value =", value)
        return value
    return p_ar_mean_unmoved

def show_bootstr_compar_mean(sample, bootstr_mass, dens_func_teor, num_points):
    fig, axes = plt.subplots(1,2)
    num_of_bins = 1 + int(math.log2(len(bootstr_mass)))
    # print("num of bins =",num_of_bins)
    # axes[0].hist(bootstr_mass, num_of_bins, edgecolor='black')

    x_min = np.min(sample)
    x_max = np.max(sample)
    x_mass = np.linspace(x_min, x_max, num=num_points)
    y_mass_teor = np.array([dens_func_teor(x) for x in x_mass])
    dens_func_samp = get_density_func(bootstr_mass)
    dens_func_samp_unmoved = get_density_func_unmoved(bootstr_mass)
    y_mass_samp = np.array([dens_func_samp(x) for x in x_mass])
    y_mass_samp_unmoved = np.array([dens_func_samp_unmoved(x) for x in x_mass])
    # axes[0].plot(x_mass, y_mass_teor,'b-', linewidth=2)

    axes[0].hist(bootstr_mass, num_of_bins, edgecolor='black', density=True)
    axes[0].plot(x_mass, y_mass_teor,'g-', linewidth=2, label='ЦПТ, теоретическая')
    axes[0].plot(x_mass, y_mass_samp,'r-', linewidth=2, label='ЦПТ, по выборке')
    axes[0].plot(x_mass, y_mass_samp_unmoved,'y--', linewidth=2, label='ЦПТ, по выборке,\n с несмещён. дисперс.')
    axes[0].set_title("Оценка плотности распред. ср. арифм. эл-тов выборки")
    axes[0].legend()
    plt.show()

# ЦПТ среднего арифметического по выборке
# поиск мат.ожидания
# def get_expectation(sample, density_func):
#     M = 0
#     for i in sample:
#         M += i * density_func
#     return M

# def get_variance(sample, density_func):
#     variance = get_expectation(sample**2, density_func) - (get_expectation(sample, density_func))**2
#     return variance


# # поиск функции плотности распределения медианы на основе выборки
# def get_density_func_median(sample):
#     mat_exp = get_ar_mean(sample) # ещё подумать над этим
#     variance = get_ar_mean(sample**2) - get_ar_mean(sample)**2
#     n = len(sample)
#     def p_median(x):
#         value = (1/((variance * 2 * np.pi)**0.5)) * np.exp(-((x - mat_exp)**2)/(2 * variance))
#         # print("value =", value)
#         return value
#     return p_median

# сравнение плотности распределения медианы выборки с бутстраповской оценкой этой плотности
# def get_density_func(sample):
#     mat_exp = get_ar_mean(sample) # ещё подумать над этим
#     variance = get_ar_mean(sample**2) - get_ar_mean(sample)**2
#     n = len(sample)
#     def p_ar_mean(x):
#         value = (1/((variance * 2 * np.pi)**0.5)) * np.exp(-((x - mat_exp)**2)/(2 * variance))
#         # print("value =", value)
#         return value
#     return p_ar_mean

def get_density_func_median(sample):
    n = len(sample)
    k = int((n - 1) / 2)
    def p_median_samp(x):
        value = (p(x) * math.factorial(n)/(math.factorial(k) * math.factorial(n - k - 1))
                * (emp_distr_func(sample, x)**k) * (1 - emp_distr_func(sample, x))**(n - k - 1))
        # print("p_median_teor =", value)
        return value
    return p_median_samp

def show_bootstr_compar_median(sample, bootstr_mass, dens_func_teor, num_points):
    fig, axes = plt.subplots(1,2)
    num_of_bins = 1 + int(math.log2(len(bootstr_mass)))
    # print("num of bins =",num_of_bins)
    # axes[0].hist(bootstr_mass, num_of_bins, edgecolor='black')

    x_min = np.min(sample)
    x_max = np.max(sample)
    x_mass = np.linspace(x_min, x_max, num=num_points)
    y_mass_teor = np.array([dens_func_teor(x) for x in x_mass])
    dens_func_samp = get_density_func_median(sample)
    y_mass_samp = np.array([dens_func_samp(x) for x in x_mass])
    # axes[0].plot(x_mass, y_mass_teor,'b-', linewidth=2)

    axes[0].hist(bootstr_mass, num_of_bins, edgecolor='black', density=True)
    axes[0].plot(x_mass, y_mass_teor,'y-', linewidth=2, label='теоретическая')
    axes[0].plot(x_mass, y_mass_samp,'r-', linewidth=2, label='по выборке')
    axes[0].set_title("Плотность распред. медианы выборки")
    axes[0].legend()
    plt.show()