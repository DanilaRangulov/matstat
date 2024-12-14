import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import norm, poisson, kstest, skew, chisquare
from scipy import stats
matplotlib.use('TkAgg')
from scipy.stats import lognorm, kstest

# Чтение CSV файла
file_path = 'chess_games.csv'  # Замените на путь к вашему файлу
data = pd.read_csv(file_path)
data['move_count'] = data['moves'].apply(lambda x: len(x.split()))
data = data['move_count']
sorted_data = np.sort(data)
n = len(data)
x = np.linspace(min(data), max(data) + 1, 1000)

params_f = stats.f.fit(data)
params_lognorm = stats.lognorm.fit(data, loc=0)
params_chi2 = stats.chi2.fit(data, loc=0)

efr = np.arange(1, n + 1) / n
cdf_theoretical_lognorm = lognorm.cdf(sorted_data, *params_lognorm)
cdf_theoretical_f = stats.f.cdf(sorted_data, *params_f)
cdf_theoretical_chi2 = stats.chi2.cdf(sorted_data, *params_chi2)

def ktest(efr, cdf_theoretical):
    K = np.max(np.abs(efr - cdf_theoretical))
    return K

print(ktest(efr, cdf_theoretical_lognorm))
print(ktest(efr, cdf_theoretical_chi2))
print(ktest(efr, cdf_theoretical_f))
print(kstest(efr, cdf_theoretical_lognorm))
print(kstest(efr, cdf_theoretical_chi2))
print(kstest(efr, cdf_theoretical_f))