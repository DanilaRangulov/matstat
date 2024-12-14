import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, kstest, mannwhitneyu, ttest_ind, shapiro, anderson, chi2
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
file_path = 'chess_games.csv'
data = pd.read_csv(file_path)

data['move_counts'] = data['moves'].apply(lambda x: len(x.split())) # X
data['rating_diff'] = (data['white_rating'] + data['black_rating']) # Y

# Корреляция Спирмена
spearman_corr = data[['rating_diff', 'move_counts']].corr(method='spearman')
# print("Корреляция Спирмена:\n", spearman_corr)
n = len(data)

#Реализация Пирсона

hat_XY = np.sum(data['move_counts'] * data['rating_diff']) / n
hat_X = np.sum(data['move_counts']) / n
hat_Y = np.sum(data['rating_diff']) / n
sigma_x = np.sqrt(np.mean(data['move_counts']**2) - np.mean(data['move_counts'])**2)
sigma_y = np.sqrt(np.mean(data['rating_diff']**2) - np.mean(data['rating_diff'])**2)
r_xy = (hat_XY - hat_Y * hat_X) / (sigma_y * sigma_x)

#Подсчет наблюдаемого значения

t_nabl = abs(r_xy) * np.sqrt((n-2) / (1 - r_xy**2))
print(t_nabl)

# df = n - 2
# alpha = 0.05
# critical_t = stats.t.ppf(1 - alpha / 2, df)
# print(critical_t)