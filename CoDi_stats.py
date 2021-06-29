from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

avg_arr = np.array([8.8,7.8,7.18,6.86,5.59,5.57,5.07,5.05,2.84,2.7])

options_arr = np.array([0.24228034876781174, 0.2979746239806308, 0.3406460721523778, 0.3616913178910738, 0.4990665188844437, 0.56004138667249, 0.5609672998432204, 0.5654079314407037,  0.7431022217527246, 0.7582531025618047])


avg_arr_horizontal = np.array([8.08,7.66,7.52,7.49,7.44,6.46,5.29,4.56,3.4,2.55])
options_arr2 = np.array([0.3300055076657796, 0.5312057173551428, 0.5418038362965045, 0.5577599021436124, 0.6156496253649152, 0.6476391737140287, 0.7554621265893472, 0.8920253989146197, 0.9723394353176292, 0.9752016707012012])
print("VERTICAL MASHUPS: PEARSON CORRELATION")
print(stats.pearsonr(avg_arr,options_arr))


result = stats.linregress(avg_arr,options_arr)
print("R-squared VERTICAL MASHUPS:"+str(result.rvalue**2))
plt.figure(1)
plt.scatter(avg_arr,options_arr, c='0.3', label="Original Vertical data")
plt.plot(avg_arr,result.intercept + result.slope*avg_arr,label='Regression line: y='+str(result.intercept)+str(result.slope)+'x, r-squared=' + str(result.rvalue**2)+', p-value=' + str(result.pvalue))
plt.legend()


print("HORIZONTAL MASHUPS: PEARSON CORRELATION")
print(stats.pearsonr(options_arr2,avg_arr_horizontal))

result_horizontal = stats.linregress(avg_arr_horizontal, options_arr2)
print("R-squared HORIZONTAL MASHUPS:"+str(result_horizontal.rvalue**2))
plt.figure(2)
plt.scatter(avg_arr_horizontal, options_arr2, c='0.3', label="Original Horizontal data")
plt.plot(avg_arr_horizontal,result_horizontal.intercept + result_horizontal.slope*avg_arr_horizontal,label='Regression line: y='+str(result_horizontal.intercept)+'+'+str(result_horizontal.slope)+'x, r-squared='+str(result_horizontal.rvalue**2)+', p-value=' + str(result_horizontal.pvalue))
plt.legend()
plt.show()