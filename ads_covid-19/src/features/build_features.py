# %load src/features/build_features.py
# Linear regression models
import numpy as np
import pandas as pd
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept = True)

from scipy import signal

def get_doubling_time_via_regression(in_array):

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1,1)

    assert len(in_array) == 3
    reg.fit(X,y)
    intercept = reg.intercept_
    slope = reg.coef_

    return intercept/slope

# Whenever the Python interpreter reads a source file, it does two things:
# it sets a few special variables like __name__, and then
# it executes all of the code found in the file.
# It's as if the interpreter inserts this at the top
# of your module when run as the main program.
# __name__ == "__main__"
#if __name__ == '__main__':
    #test_data = np.array([2,4,6])
    #result = get_doubling_time_via_regression(test_data)
    #print('The test slope is: ' + str(result))

def savgol_filter(df_input, column ='confirmed', window = 5):
    ''' Filter the data'''
    degree = 1
    df_result = df_input

    filter_in = df_input[column].fillna(0) # Fill NA/NaN values using the specified method

    result = signal.savgol_filter(np.array(filter_in),
                                 window,
                                 degree)
    df_result[str(column+'_filtered')] = result
    return df_result

def rolling_reg(df_input,col='confirmed'):
    ''' Rolling Regression to approximate the doubling time'''
    ''' Connected to -> get_doubling_time_via_regression'''
    days_back = 3
    result = df_input[col].rolling(
                window = days_back,
                min_periods = days_back).apply(get_doubling_time_via_regression, raw = False)
    return result

def calc_filtered_data(df_input, filter_on = 'confirmed'):
    ''' This function does all the merging of the new filtered data'''
    ''' Connected to -> savgol_filter'''
    # Set creates an unordered list of the given parameters
    must_contain = set(['state', 'country', filter_on])
    # Asserting whether state and country included in df_input
    assert must_contain.issubset(set(df_input.columns)), 'comment after a comma is accepted'

    df_output = df_input.copy()
    pd_filtered_result = df_output[['state', 'country', filter_on]].groupby(['state', 'country']).apply(savgol_filter)
    df_output = pd.merge(df_output, pd_filtered_result[[str(filter_on + '_filtered')]], left_index = True, right_index = True, how = 'left')
    return df_output.copy()

def calc_doubling_rate(df_input, filter_on = 'confirmed'):
    ''' Connected to -> rolling_reg'''
    must_contain = set(['state', 'country', filter_on])
    assert must_contain.issubset(set(df_input.columns)), 'comment after a comma is accepted'

    # Apply rolling_reg to the column 'confirmed' on states of countries
    pd_DR_result = df_input.groupby(['state', 'country']).apply(rolling_reg, filter_on).reset_index()

    pd_DR_result = pd_DR_result.rename(columns = {filter_on : filter_on+'_DR', 'level_2' :'index'})

    df_output = pd.merge(df_input, pd_DR_result[['index', str(filter_on + '_DR')]], left_index = True, right_on = ['index'], how = 'left')
    df_output = df_output.drop(columns = ['index'])

    return df_output

if __name__ == '__main__':
    test_data_reg = np.array([2,4,6])
    result = get_doubling_time_via_regression(test_data_reg)
    print('The test slope is: ' + str(result))

    pd_JH_data = pd.read_csv('data/processed/COVID_relational_confirmed.csv', sep = ';', parse_dates = [0])
    pd_JH_data = pd_JH_data.sort_values('date', ascending = True).copy()

    pd_result_larg = calc_filtered_data(pd_JH_data)
    pd_result_larg = calc_doubling_rate(pd_result_larg)
    # overwrites the parameter 'filter_on'
    pd_result_larg = calc_doubling_rate(pd_result_larg, 'confirmed_filtered')

    mask = pd_result_larg['confirmed'] > 100
    pd_result_larg['confirmed_filtered_DR'] = pd_result_larg['confirmed_filtered_DR'].where(mask, other = np.NaN)
    pd_result_larg.to_csv('data/processed/COVID_final_set.csv', sep = ';', index = False)
    print(pd_result_larg[pd_result_larg['country'] == 'Germany'].tail())
