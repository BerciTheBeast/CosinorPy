import pandas as pd
import numpy as np
np.seterr(divide='ignore')
import statsmodels.api as sm
import statsmodels.tsa as smt
import statsmodels.stats.multitest as multi
import math
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import brute
from statsmodels.tsa.arima.model import ARIMA
from CosinorPy import evaluate_rhythm_params

from CosinorPy.helpers import df_add_row

def remove_polin_comp_df(df, degree = 3, period = 24, summary_file=""):
    """
    Attempts to remove the polynomial component from the provided CosinorPy-format dataframe
    """
    df2 = pd.DataFrame(columns=df.columns)

    if summary_file:
        df_fit = pd.DataFrame(columns=['test', 'k', 'CI', 'p', 'q'])

    for test in df.test.unique():
        x,y = df[df['test']==test].x,df[df['test']==test].y
        x,y,fit = remove_polin_comp(x,y,degree=degree, period=period, return_fit=True)
        df_tmp = pd.DataFrame(columns=df.columns)
        df_tmp['x'] = x
        df_tmp['y'] = y
        df_tmp['test'] = test
        # df2 = df2.append(df_tmp, ignore_index=True)
        df2 = pd.concat([df2, df_tmp], ignore_index=True)
        if summary_file:
            fit['test'] = test
            # df_fit=df_fit.append(fit, ignore_index=True)
            df_fit = df_add_row(df_fit, fit)
    if summary_file:
        df_fit.q = multi.multipletests(df_fit.p, method = 'fdr_bh')[1]
        if summary_file.endswith("csv"):
            df_fit.to_csv(summary_file, index=False)
        elif summary_file.endswith("xlsx"):
            df_fit.to_excel(summary_file, index=False)
    return df2
    
# performs detrending only if linear model is significant
def remove_polin_comp(X, Y, degree = 3, period = 24, return_fit=False):
    """
    Attempts to remove the polynomial component from the provided X and Y arrays.
    
    Using OLS (ordinary least squares) this functions fits a `degree`-degree polynomial onto the data, then subtracts the polinomyal prediction from the Y data.
    """
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)

    polynomial_features = PolynomialFeatures(degree=degree)
    X_fit = polynomial_features.fit_transform(X.reshape(-1, 1))
    model = sm.OLS(Y, X_fit)
    results = model.fit()

    CIs = results.conf_int()
    if type(CIs) != np.ndarray:
        CIs = CIs.values
    CI = CIs[1]
    
    #A = results.params[0]
    k = results.params[1]


    """      
    X_lin = np.zeros(X_fit.shape)
    X_lin[:,1] = X_fit[:,1]
    Y_lin = results.predict(X_lin)
    Y = Y-Y_lin
    """
    #Y_fit = results.predict(X_fir)
    #Y = Y - Y_fit

    
    
    #Y = Y - A - k*X
    if CI[0] * CI[1] > 0: # if both CIs have the same sign
        Y_pred = results.predict(X_fit).reshape(-1, 1)
        Y = Y - Y_pred
    
    if return_fit:
        fit = {}
        fit['k'] = results.params[1]
        fit['CI'] = CI
        fit['p'] = results.pvalues[1]
        fit['A'] = results.params[0]

        return X,Y,fit    
    """
    X_fit = generate_independents(X, n_components = n_components, period = period, lin_comp = False)
    model = sm.OLS(Y, X_fit)
    results = model.fit()
    plt.plot(X, results.fittedvalues, color="black")
    """
    
    return X, Y

def arima_objfunc(order, exog, endog):
    model = ARIMA(endog, exog, order)
    model.initialize_approximate_diffuse()
    fit = model.fit()
    return fit.aic

# Arima is a fit, but others (amp & abs min max are pure mathematical manipulation, i think)
def remove_arima_comp_df(df, period = 24, summary_file=""):
    """
    Attempts to detrend the provided CosinorPy-format dataframe using ARIMA.
    """
    df2 = pd.DataFrame(columns=df.columns)

    if summary_file:
        df_fit = pd.DataFrame(columns=['test', 'k', 'CI', 'p', 'q'])

    for test in df.test.unique():
        x,y = df[df['test']==test].x,df[df['test']==test].y
        x,y,fit = remove_arima_comp(x,y, period=period, return_fit=True)
        df_tmp = pd.DataFrame(columns=df.columns)
        df_tmp['x'] = x
        df_tmp['y'] = y
        df_tmp['test'] = test
        # df2 = df2.append(df_tmp, ignore_index=True)
        df2 = pd.concat([df2, df_tmp], ignore_index=True)
        if summary_file:
            fit['test'] = test
            # df_fit=df_fit.append(fit, ignore_index=True)
            df_fit = df_add_row(df_fit, fit)
    if summary_file:
        df_fit.q = multi.multipletests(df_fit.p, method = 'fdr_bh')[1]
        if summary_file.endswith("csv"):
            df_fit.to_csv(summary_file, index=False)
        elif summary_file.endswith("xlsx"):
            df_fit.to_excel(summary_file, index=False)
    
    return df2
    
# performs detrending only if linear model is significant
def remove_arima_comp(X, Y_df, period = 24, return_fit=False):
    """
    Attempts to detrend the provided X and Y arrays using ARIMA.

    Using a grid search of the 3 ARIMA parameters we find a model that fits, then subtract it from the Y data.
    """
    
    X = np.array(X)
    Y = np.array(Y_df)

    
    grid = (slice(0, 3, 1), slice(0, 3, 1), slice(0, 3, 1))
    order = brute(arima_objfunc, grid, args=(X, Y), finish=None)

    # model = smt.arima.model.ARIMA(Y, order=order, trend='t')
    try:
        model = smt.arima.model.ARIMA(Y, order=order, trend=[0, 1, 1, 1])
        model.initialize_approximate_diffuse()
        results = model.fit()
    except:
        model = smt.arima.model.ARIMA(Y, order=order, trend=[0, 0, 1, 1])
        model.initialize_approximate_diffuse()
        results = model.fit()

    CIs = results.conf_int()
    if type(CIs) != np.ndarray:
        CIs = CIs.values
    CI = CIs[1]
    
    """      
    X_lin = np.zeros(X_fit.shape)
    X_lin[:,1] = X_fit[:,1]
    Y_lin = results.predict(X_lin)
    Y = Y-Y_lin
    """
    #Y_fit = results.predict(X_fir)
    #Y = Y - Y_fit

    
    
    #Y = Y - A - k*X
    if CI[0] * CI[1] > 0: # if both CIs have the same sign
        Y_pred = results.predict()
        Y = Y - Y_pred
        # Y = Y - k*X
    
    if return_fit:
        fit = {}
        fit['k'] = results.params[1]
        fit['CI'] = CI
        fit['p'] = results.pvalues[1]
        fit['A'] = results.params[0]

        return X,Y,fit    
    """
    X_fit = generate_independents(X, n_components = n_components, period = period, lin_comp = False)
    model = sm.OLS(Y, X_fit)
    results = model.fit()
    plt.plot(X, results.fittedvalues, color="black")
    """
    
    return X, Y

def gauss(n=11,sigma=1):
    """
    Generates a gauss filter.
    """
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]

def remove_amp_comp_df(df, phase=None, period=24, summary_file=""):
    """
    Attempts to remove an amplitude trend from the provided CosinorPy-format dataframe
    """
    df2 = pd.DataFrame(columns=df.columns)

    if summary_file:
        df_fit = pd.DataFrame(columns=['test', 'k', 'CI', 'p', 'q'])

    for test in df.test.unique():
        x,y = df[df['test']==test].x,df[df['test']==test].y
        x,y,fit = remove_amp_comp(x,y, phase=phase, period=period, return_fit=True)
        df_tmp = pd.DataFrame(columns=df.columns)
        df_tmp['x'] = x
        df_tmp['y'] = y
        df_tmp['test'] = test
        # df2 = df2.append(df_tmp, ignore_index=True)
        df2 = pd.concat([df2, df_tmp], ignore_index=True)
        if summary_file:
            fit['test'] = test
            # df_fit=df_fit.append(fit, ignore_index=True)
            df_fit = df_add_row(df_fit, fit)
    if summary_file:
        df_fit.q = multi.multipletests(df_fit.p, method = 'fdr_bh')[1]
        if summary_file.endswith("csv"):
            df_fit.to_csv(summary_file, index=False)
        elif summary_file.endswith("xlsx"):
            df_fit.to_excel(summary_file, index=False)
    
    return df2

def remove_amp_comp(X, Y, phase=None, period = 24, return_fit=False):
    """
    Attempts to remove an amplitude trend from the provided X and Y arrays.

    The function obtains rhythm params for all intervals of the data, then multiply the Y data to bring the amplitude of all intervals to the average of all amplitudes of intervals.
    """
    X = np.array(X)
    Y = np.array(Y)
    
    params = evaluate_rhythm_params(X, Y, period = period, phase=phase)
    Y_new = detrend_amp(X, Y, period = period, phaseLoc=params['max_loc'], amps=params['interval_stats_obj']['AMPLITUDE_per'])
    
    if return_fit:
        fit = {}
        # fit['k'] = results.params[1]
        # fit['CI'] = CI
        # fit['p'] = results.pvalues[1]
        # fit['A'] = results.params[0]

        return X,Y_new,fit    
    
    return X, Y_new


def detrend_amp(X, Y, period, phaseLoc, amps=[]):
    """
    loop through X, get indices, then correct Y via multiplication. Feed into new Y with same dimensions as old Y
    """
    # as above, so below. loop through X, get indices, then correct Y via multiplication. Feed into new Y with same dimensions as old Y
    Y_new = []

    if (len(amps) == 0 or np.average(amps) < 1e-5): # so we don't end up with too low numbers
        mult_factor = [1]
    else:
        mult_factor = np.average(amps) / (np.array(amps) + 1e-16) # so we avoid zero division
    
    if (0 != phaseLoc):
        Y_new.append(Y[0:phaseLoc])
    
    startLoc = phaseLoc
    curr_apm_idx = 0
    while True:
        endLocs = np.where(X > (X[min(startLoc, len(X))] + period))
    
        if (len(endLocs) > 0 and len(endLocs[0]) > 0):
            endLoc = endLocs[0][0]
            Y_new.append(Y[startLoc:endLoc] * mult_factor[curr_apm_idx])
    
            startLoc = endLoc
        else:
            
    
            if (startLoc != len(X)):
                Y_new.append(Y[startLoc:len(X)] * mult_factor[curr_apm_idx])
    
            break
        if(curr_apm_idx + 1 < len(mult_factor)):
            curr_apm_idx += 1
    
    
    return np.concatenate(Y_new)

def remove_z_score_comp_df(df, summary_file=""):
    """
    Attempts to remove trends from the provided CosinorPy-format dataframe by computing the Z-score of the data.
    """
    df2 = pd.DataFrame(columns=df.columns)

    if summary_file:
        df_fit = pd.DataFrame(columns=['test', 'k', 'CI', 'p', 'q'])

    for test in df.test.unique():
        x,y = df[df['test']==test].x,df[df['test']==test].y
        x,y,fit = remove_z_score_comp(x,y, return_fit=True)
        df_tmp = pd.DataFrame(columns=df.columns)
        df_tmp['x'] = x
        df_tmp['y'] = y
        df_tmp['test'] = test
        # df2 = df2.append(df_tmp, ignore_index=True)
        df2 = pd.concat([df2, df_tmp], ignore_index=True)
        if summary_file:
            fit['test'] = test
            # df_fit=df_fit.append(fit, ignore_index=True)
            df_fit = df_add_row(df_fit, fit)
    if summary_file:
        df_fit.q = multi.multipletests(df_fit.p, method = 'fdr_bh')[1]
        if summary_file.endswith("csv"):
            df_fit.to_csv(summary_file, index=False)
        elif summary_file.endswith("xlsx"):
            df_fit.to_excel(summary_file, index=False)
    
    return df2

def remove_z_score_comp(X, Y, period = 24, return_fit=False):
    """
    Attempts to remove trends from the provided X and Y arrays by computing the Z-score of the data.
    """
    X = np.array(X)
    Y = np.array(Y)

    params = evaluate_rhythm_params(X, Y, period = period)

    Y_new = (Y - params['mean']) / (params['std'] + 1e-8) # so no division by 0

    
    if return_fit:
        fit = {}
        # fit['k'] = results.params[1]
        # fit['CI'] = CI
        # fit['p'] = results.pvalues[1]
        # fit['A'] = results.params[0]

        return X,Y_new,fit    
    
    return X, Y_new


def remove_min_max_abs_comp_df(df, scale_amp=False, summary_file=""):
    """
    Attempts to remove trends from the provided CosinorPy-format dataframe by computing the Z-score of the data.
    """
    df2 = pd.DataFrame(columns=df.columns)

    if summary_file:
        df_fit = pd.DataFrame(columns=['test', 'k', 'CI', 'p', 'q'])

    for test in df.test.unique():
        x,y = df[df['test']==test].x,df[df['test']==test].y
        x,y,fit = remove_min_max_abs_comp(x,y,return_fit=True,scale_amp=scale_amp)
        df_tmp = pd.DataFrame(columns=df.columns)
        df_tmp['x'] = x
        df_tmp['y'] = y
        df_tmp['test'] = test
        # df2 = df2.append(df_tmp, ignore_index=True)
        df2 = pd.concat([df2, df_tmp], ignore_index=True)
        if summary_file:
            fit['test'] = test
            # df_fit=df_fit.append(fit, ignore_index=True)
            df_fit = df_add_row(df_fit, fit)
    if summary_file:
        df_fit.q = multi.multipletests(df_fit.p, method = 'fdr_bh')[1]
        if summary_file.endswith("csv"):
            df_fit.to_csv(summary_file, index=False)
        elif summary_file.endswith("xlsx"):
            df_fit.to_excel(summary_file, index=False)
    
    return df2


# Appears to somewhat mitigate linear trends at a severe cost to amplitude
def remove_min_max_abs_comp(X, Y, return_fit=False, scale_amp = False):
    """
    Attempts to detrend data by dividing the data with the maximum of the abosolutes of the minimum and maximum values of the data
    """
    
    m = min(Y)
    M = max(Y)
    abs_m = abs(m)
    abs_M = abs(M)
    amp = abs(M - m)

    # Y_new = Y / abs_m / abs_M
    Y_new = Y / max(abs_m, abs_M)

    if scale_amp:
        Y_new = amp * Y_new

    if return_fit:
        fit = {}
        return X,Y_new,fit

    return X, Y_new


def remove_baseline_comp_df(df, width=7, summary_file=""):
    """
    Attempts to detrend the provided data using smoothing.

    It smoothes the provided data, then subtracts it from Y and returns the result.
    """
    df2 = pd.DataFrame(columns=df.columns)
    if summary_file:
        df_fit = pd.DataFrame(columns=['test', 'k', 'CI', 'p', 'q'])

    for test in df.test.unique():
        x,y = df[df['test']==test].x,df[df['test']==test].y
        x,y,fit = remove_baseline_comp(x,y,width=width, return_fit=True)
        df_tmp = pd.DataFrame(columns=df.columns)
        df_tmp['x'] = x
        df_tmp['y'] = y
        df_tmp['test'] = test
        # df2 = df2.append(df_tmp, ignore_index=True)
        df2 = pd.concat([df2, df_tmp], ignore_index=True)
        if summary_file:
            fit['test'] = test
            # df_fit=df_fit.append(fit, ignore_index=True)
            df_fit = df_add_row(df_fit, fit)
    if summary_file:
        df_fit.q = multi.multipletests(df_fit.p, method = 'fdr_bh')[1]
        if summary_file.endswith("csv"):
            df_fit.to_csv(summary_file, index=False)
        elif summary_file.endswith("xlsx"):
            df_fit.to_excel(summary_file, index=False)
    
    return df2

# performs detrending via moving average
def remove_baseline_comp(X, Y, width = 7, return_fit=False):
    """
    Attempts to detrend the provided data using smoothing.

    It smoothes the provided data, then subtracts it from Y and returns the result.
    """

    X = np.array(X)
    Y = np.array(Y)
    kernel = gauss(width)
    fit = np.convolve(Y, kernel, 'same') # same size, but boundary effects at edges

    # remove averaged value    
    Y = Y - fit
    
    if return_fit:
        fit = {}
        # fit['k'] = results.params[1]
        # fit['CI'] = CI
        # fit['p'] = results.pvalues[1]
        # fit['A'] = results.params[0]

        return X,Y,fit    
    
    return X, Y



# computes scores with provided functions on tests in dfs
def compare_graphs_df(df_one, df_two, scores_functions):
    """
    computes scores with provided functions on tests in dfs
    """
    scores_obj = {}
    for test in df_one.test.unique():
        x_one,y_one = df_one[df_one['test']==test].x,df_one[df_one['test']==test].y
        x_two,y_two = df_two[df_two['test']==test].x,df_two[df_two['test']==test].y
        scores_obj_tmp = compare_graphs(y_one,y_two, scores_functions)
        scores_obj[test] = scores_obj_tmp
    return scores_obj

def compare_graphs(Y_one, Y_two, scores_functions=[]):
    """
    computes scores with provided functions on provided dfs
    """
    scores_list = []
    for score in scores_functions:
        score_val = score(Y_one, Y_two)
        scores_list.append(score_val)
    return scores_list



