import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, kstest, mannwhitneyu, ttest_ind, shapiro, anderson, chi2
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

file_path = 'chess_games.csv'
data_df = pd.read_csv(file_path)

data_df['move_count'] = data_df['moves'].apply(lambda x: len(x.split()))


rated_games = data_df[data_df['rated'] == True]['move_count'].values
unrated_games = data_df[data_df['rated'] == False]['move_count'].values


unique_move_counts = np.union1d(np.unique(rated_games), np.unique(unrated_games))


nu = np.zeros((2, len(unique_move_counts)), dtype=int)


for i, move_count in enumerate(unique_move_counts):
    nu[0, i] = np.sum(rated_games == move_count)


for i, move_count in enumerate(unique_move_counts):
    nu[1, i] = np.sum(unrated_games == move_count)

hat_p = np.sum(nu, axis=0) / np.sum(nu)
chi2_score = 0
for i in range(2):
  for j in range(4):
    chi2_score += np.power(nu[i,j] - np.sum(nu, axis=1)[i] * hat_p[j], 2) / (np.sum(nu, axis=1)[i] * hat_p[j])
sign_level = 0.05
print(f"chi2_score = {chi2_score}")
print(f"significance level = {sign_level}")
p_value = 1 - chi2.cdf(chi2_score, df=1 * 3)
print(f"p_value = {p_value}")