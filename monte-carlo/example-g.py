#Using Monte Carlo method to estimate g from the model g=2*y/t^2 and measurement data for y and t from experiment. 

# model is g=2y/t^2 + error 

# y and t come from data. they come with noise. 

#Method: Random Monte carlo search 

#1- repeatedly draw random trials models (gtrial)  from 0 to 20. 

# 2 for each gtrial, calculate the predicted y values using the model and the t values from the data.
#3- calculate predicted error for each gtrial as the sum of squared differences between the predicted y values and the actual y values from the data.

#4- if the trial has a lower rror than the best solution found so far, accept it as a the new best model. 

import numpy as np 
gmin = 0
gmax = 20
num_trials = 10000
best_g = None
best_error = float('inf')

for i in range(num_trials):
    g_trial = np.random.uniform(gmin, gmax)
    y_predicted = 0.5 * g_trial * t**2
    error = np.sum((y - y_predicted)**2)

    if error < best_error:
        best_error = error
        best_g = g_trial

print(f'Estimated g from Monte Carlo: {best_g:.2f}')



