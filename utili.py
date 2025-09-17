# utilities.py

import yfinance as yf
import numpy as np
from scipy import stats

def get_risk_free_rate():
    """Fetch risk-free rate with multiple fallback options"""
    try:
        treasury = yf.Ticker("^TNX")
        treasury_data = treasury.history(period="1d")
        return treasury_data['Close'].iloc[-1] / 100
    except:
        try:
            # Alternative treasury ticker
            treasury = yf.Ticker("^IRX")
            treasury_data = treasury.history(period="1d")
            return treasury_data['Close'].iloc[-1] / 100
        except:
            print("Warning: Using fallback risk-free rate of 2%")
            return 0.02


def estimate_dividend_yield(ticker, S0):
    """More robust dividend yield estimation"""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Try multiple approaches to get dividend yield
    for key in ['dividendYield', 'trailingAnnualDividendYield', 'forwardAnnualDividendYield']:
        if key in info and info[key] is not None:
            yield_value = info[key]
            if yield_value > 0.1:  # if yield is above 10%, it might be error
                return 0.0
            return yield_value
    
    # If no dividend yield found, calculate from recent dividends
    try:
        dividends = stock.dividends
        if len(dividends) > 0:
            annual_dividend = dividends.tail(4).sum()
            current_price = info.get('currentPrice', S0)
            return annual_dividend / current_price
    except:
        pass
    
    return 0.0


def log_likelihood(params, log_vol, dt=1/252):
    k, theta, x = params
    if k <= 0 or theta <= 0 or x <= 0:
        return np.inf
    mu_ou = log_vol[:-1] * np.exp(-k * dt) + np.log(theta) * (1 - np.exp(-k * dt))
    sigma_ou = x * np.sqrt((1 - np.exp(-2 * k * dt)) / (2 * k))
    ll = np.sum(stats.norm.logpdf(log_vol[1:], loc=mu_ou, scale=sigma_ou))
    return -ll if np.isfinite(ll) else np.inf
