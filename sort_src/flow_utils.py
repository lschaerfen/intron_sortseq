import flowkit as fk
import polars as pl
import numpy as np
from scipy.optimize import curve_fit
import datetime as dt

def load_flow(filename):

    data = fk.Sample(filename)
    time = dt.datetime.strptime(data.get_metadata()['etim'], '%H:%M:%S').time()
    seconds = time.hour * 3600 + time.minute * 60 + time.second # Convert time to seconds
    data = fk.Sample(filename)
    data = data.as_dataframe(source='raw')
    data.columns = data.columns.droplevel(1)

    # convert to polars
    data = pl.from_pandas(data)

    # remove negative values
    data = data.select([
        pl.when(pl.col(col) > 0)
        .then(pl.col(col))
        .otherwise(np.nan)
        .alias(col)
        for col in data.columns
    ])

    return(data, seconds)

def fit_gauss(rat, bins, n=1, guess=None, bounds=None):
    
    y, x = np.histogram(rat, bins=bins, density=True)
    x = x[:-1]+0.5*np.diff(x)[0]

    if n == 1:

        if guess is None:
            guess = [np.mean(rat), np.std(rat)]
        (mean_hat, std_hat), _ = curve_fit(gauss, x, y, p0=guess)
        y_hat = gauss(x, mean_hat, std_hat)
        
        # residual sum of squares
        ss_res = np.sum((y - y_hat) ** 2)
        # total sum of squares
        ss_tot = np.sum((y - np.mean(y_hat)) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        
        return(x, y, y_hat, mean_hat, std_hat, r2)

    if n == 2:

        if guess is None:
            guess = [0.5, np.mean(rat), np.std(rat), np.mean(rat), np.std(rat)]
        if bounds is None:
            bounds = ([0, -np.inf, -np.inf, -np.inf, -np.inf], [1, np.inf, np.inf, np.inf, np.inf])

        (amp1_hat, mean1_hat, std1_hat, mean2_hat, std2_hat), _ = curve_fit(gauss_2, x, y, p0=guess, bounds=bounds)
        y_hat = gauss_2(x, amp1_hat, mean1_hat, std1_hat, mean2_hat, std2_hat)
        
        # residual sum of squares
        ss_res = np.sum((y - y_hat) ** 2)
        # total sum of squares
        ss_tot = np.sum((y - np.mean(y_hat)) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        
        return(x, y, y_hat, amp1_hat, mean1_hat, std1_hat, mean2_hat, std2_hat, r2)
    
def gauss(x, mean, std):
    return (1/(std*np.sqrt(2*np.pi)))*np.exp(-(x-mean)**2/(2*std**2))

def gauss_2(x, amp1, mean1, std1, mean2, std2):
    return amp1*gauss(x, mean1, std1) + (1-amp1)*gauss(x, mean2, std2)