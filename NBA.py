import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None)    #shows all rows
pd.set_option('display.width', 1000)         # increase total width
pd.set_option('display.expand_frame_repr', False)  # prevent wrapping
np.set_printoptions(suppress=True, precision = 2)

nba = pd.read_csv('nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

#task 1. Create two series using 2010 data
knicks_pts_10 = nba_2010[nba_2010.fran_id == 'Knicks']
nets_pts_10 = nba_2010[nba_2010.fran_id == 'Nets']

#2) Calculate the diffrence between the averages of two teams score

knicks_mean_10 = np.mean(knicks_pts_10.pts)
nets_mean_10 = np.mean(nets_pts_10.pts)

diff_means_2010 = knicks_mean_10 - nets_mean_10

# print(diff_means_2010) Shows that the knicks in 2010 had an almost 10 points more on average than the nets

# plt.hist(knicks_pts_10.pts, alpha=0.8, density = True, label='knicks_10')
# plt.hist(nets_pts_10.pts, alpha=0.8, density = True, label='nets_10')
# plt.legend()
# plt.title("2010 Season")
# plt.show()
knicks_pts_14 = nba_2014[nba_2014.fran_id == 'Knicks']
nets_pts_14 = nba_2014[nba_2014.fran_id == 'Nets']

knicks_mean_14 = np.mean(knicks_pts_14.pts)
nets_mean_14 = np.mean(nets_pts_14.pts)

diff_means_2014 = knicks_mean_14 - nets_mean_14

# print(diff_means_2014) This shows that the point diffrence now is only 0.4

plt.hist(knicks_pts_14.pts, alpha=0.8, density = True, label='knicks_14')
plt.hist(nets_pts_14.pts, alpha=0.8, density = True, label='nets_14')
# plt.legend()
plt.title("2014 Season")
# plt.show() #In 2010, the Knicks and Nets had a
# larger difference in average points scored, showing
# that the teams performed differently offensively. By 2014, the
# distributions became closer together, suggesting that the Nets improved relative
# to the Knicks. The Knicks’ average scoring appears to have slightly decreased while the Nets’
# scoring became more competitive. If this trend continued in future seasons,
# the Nets could potentially outperform the Knicks in scoring.

knicks = nba[nba.fran_id == 'Knicks']
nets =nba[nba.fran_id == 'nets']

# plot(x,y)
# plt.show()

#Going to try to figure out how to group the teams by year and then make it into a mean for the amount of points

# print(knicks_grouped_year)
# knicks_years = knicks.year_id.unique()
# knicks_group = []
# for i in knicks_years:
#     knicks_year_group = knicks[knicks.year_id == i]
#     temp_mean = round(np.mean(knicks_year_group.pts),2)
#     knicks_group.append(temp_mean)
#
#
# plt.plot(knicks_years, knicks_group)
# plt.xlim(knicks_years[0] - 5)
# plt.show()
# print(knicks_years)

location_result_freq = pd.crosstab(nba_2010.game_result,nba_2010.game_location)
print(location_result_freq)

location_result_proportions = location_result_freq/len(nba_2010)
print(location_result_proportions)

chi2, pval,dof,expected = chi2_contingency(location_result_freq)
print(chi2)
print(pval)
print(expected)
print(dof)
#pval is under 0.05 showing an asscotion with the addtion of the chi2 value being higher than four and only havign one dgreee of freedom

point_diff_forecast_cov = np.cov(nba_2010.forecast,nba_2010.point_diff)
print(point_diff_forecast_cov)
#There is a strong positive linear relationship

point_diff_forecast_corr = pearsonr(nba_2010.forecast,nba_2010.point_diff)
print(point_diff_forecast_corr)
#The r value shows that the strength is modirtately postive

plt.scatter('forecast','point_diff',data = nba_2010)
plt.xlabel('forecast')
plt.ylabel('point_diff')
plt.show()
