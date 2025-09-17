import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy import stats
import warnings
import math
from scipy.stats import norm
warnings.filterwarnings('ignore')

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

# --------------------------
# User inputs
# --------------------------
ticker = input("Enter stock ticker : ").strip().upper()
N = int(input("Number of time steps for binomial tree: \n"))

# --------------------------
# Download historical data
# --------------------------
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
start_date = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')

try:
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

S0 = float(data['Close'].iloc[-1])
prices = data['Close']
print(f"Current {ticker} price: ${S0:.2f}\n")

# --------------------------
# Log-OU Calibration
# --------------------------
log_returns = np.log(prices / prices.shift(1)).dropna()
volatility_historical = log_returns.rolling(window=252).std() * np.sqrt(252)
variance_historical = (volatility_historical**2).ffill().dropna()
log_variance = np.log(variance_historical.clip(lower=1e-12).values)

def log_likelihood(params, log_vol, dt=1/252):
    k, theta, x = params
    if k <= 0 or theta <= 0 or x <= 0:
        return np.inf
    mu_ou = log_vol[:-1] * np.exp(-k * dt) + np.log(theta) * (1 - np.exp(-k * dt))
    sigma_ou = x * np.sqrt((1 - np.exp(-2 * k * dt)) / (2 * k))
    ll = np.sum(stats.norm.logpdf(log_vol[1:], loc=mu_ou, scale=sigma_ou))
    return -ll if np.isfinite(ll) else np.inf

bounds = [(1e-3, 10), (1e-3, 0.5), (1e-3, 0.5)]
result = differential_evolution(log_likelihood, bounds, args=(log_variance,), maxiter=1000, seed=42)
k, theta, x = result.x
v0 = float(variance_historical.iloc[-1])

print(f"Calibrated Log-OU parameters: k={k:.4f}, theta={theta:.4f}, x={x:.4f}")
print(f"Calibrated annual volatility: {np.sqrt(theta)*100:.2f}%")
print(f"Current annual volatility: {np.sqrt(v0)*100:.2f}%")

# --------------------------
# Option selection
# --------------------------
stock = yf.Ticker(ticker)
exp_dates = stock.options
if not exp_dates:
    raise ValueError(f"No option data for {ticker}")

print("\nAvailable expiration dates:")
for i, date in enumerate(exp_dates):
    print(f"{i+1}. {date}")
    
try:
    exp_choice = int(input("Select expiration date by number: ")) - 1
    exp_date = exp_dates[exp_choice]
except (ValueError, IndexError):
    print("Invalid selection. Using first available expiration date.")
    exp_date = exp_dates[0]

T = max((pd.Timestamp(exp_date) - pd.Timestamp.now()).total_seconds() / (365.25*24*3600), 0.0)
print(f"Time to expiration: {T:.4f} years")

r = get_risk_free_rate()
q = estimate_dividend_yield(ticker, S0)

print(f"Using risk-free rate: {r:.4f}, dividend yield: {q:.4f}")

options = stock.option_chain(exp_date)
calls = options.calls
valid_calls = calls[((calls['bid'] > 0) | (calls['ask'] > 0))].copy()
valid_calls['midPrice'] = (valid_calls['bid'] + valid_calls['ask'])/2
valid_calls['moneyness'] = valid_calls['strike']/S0
valid_calls['distance'] = abs(valid_calls['moneyness'] - 1)
# Sort by distance to moneyness to show nearest strikes first
valid_calls_sorted = valid_calls.sort_values('distance')
valid_calls_display = valid_calls_sorted.reset_index(drop=True)
valid_calls_display['Index'] = valid_calls_display.index

display_cols = ['Index','strike','bid','ask','midPrice','moneyness']
if 'impliedVolatility' in valid_calls_display.columns:
    display_cols.append('impliedVolatility')
    
print(valid_calls_display[display_cols].head(10))

print("\nOptions:\n1. Select from the list above\n2. Enter a custom strike price")
choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    num_options = min(10, len(valid_calls_display))
    try:
        strike_choice = int(input(f"Enter the index of the strike price you want to use (0-{num_options-1}): "))
        call_option = valid_calls_display.iloc[strike_choice]
        K = call_option['strike']
        market_price = call_option['midPrice']
    except (ValueError, IndexError):
        print("Invalid selection. Using first available strike.")
        call_option = valid_calls_display.iloc[0]
        K = call_option['strike']
        market_price = call_option['midPrice']
else:
    try:
        K = float(input("Enter the custom strike price: "))
        closest_idx = (valid_calls['strike'] - K).abs().idxmin()
        if abs(valid_calls.loc[closest_idx, 'strike'] - K)/K < 0.05:
            market_price = valid_calls.loc[closest_idx, 'midPrice']
        else:
            # Calculate theoretical price using Black-Scholes as fallback
            d1 = (np.log(S0/K) + (r - q + 0.5 * theta) * T) / (np.sqrt(theta) * np.sqrt(T))
            d2 = d1 - np.sqrt(theta) * np.sqrt(T)
            market_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    except ValueError:
        print("Invalid strike price. Using at-the-money strike.")
        K = S0
        closest_idx = (valid_calls['strike'] - K).abs().idxmin()
        market_price = valid_calls.loc[closest_idx, 'midPrice']

# Warn if strike is too far from current price
if abs(K - S0) / S0 > 0.5:
    print(f"Warning: Strike price ${K} is significantly different from current price ${S0}. Results may be unreliable.")

eps = 1e-12
dt = T/N if N>0 else 0.0
m = 21  # number of Markov states
n_std = 4.0
mu_stationary = np.log(theta)
var_stationary = (x**2)/(2*k) if k>0 else (x**2)*dt
logv0 = np.log(max(v0, eps))

# Create log-volatility grid
center = mu_stationary
halfwidth = n_std * math.sqrt(max(var_stationary, eps))
grid_min = min(center-halfwidth, logv0-halfwidth/2)
grid_max = max(center+halfwidth, logv0+halfwidth/2)
X_grid = np.linspace(grid_min, grid_max, m)

# Midpoints for transition probabilities
midpoints = np.zeros(m+1)
midpoints[1:-1] = 0.5*(X_grid[:-1]+X_grid[1:])
midpoints[0] = X_grid[0] - (X_grid[1]-X_grid[0])/2
midpoints[-1] = X_grid[-1] + (X_grid[-1]-X_grid[-2])

# Transition matrix P
P = np.zeros((m,m))
sigma_cond = x * math.sqrt((1 - np.exp(-2*k*dt))/(2*k)) if k>0 else x*math.sqrt(dt)
exp_m = np.exp(-k*dt) if k>0 else 1.0 - k*dt
for i in range(m):
    mu_cond = exp_m*X_grid[i] + (1-exp_m)*np.log(theta)
    cdf_vals = norm.cdf((midpoints - mu_cond)/max(sigma_cond, 1e-16))
    probs = np.clip(cdf_vals[1:] - cdf_vals[:-1], 0.0, None)
    s = probs.sum()
    if s <= 0:
        idx = int(np.argmin(np.abs(X_grid - mu_cond)))
        probs = np.zeros_like(probs)
        probs[idx] = 1.0
    else:
        probs /= s
    P[i,:] = probs

# Initial state
state0 = int(np.argmin(np.abs(X_grid - logv0)))

# Distribution over Markov states at each time step
dist = np.zeros((N+1, m))
dist[0, state0] = 1.0
for t in range(1, N+1):
    dist[t,:] = dist[t-1,:] @ P

# Expected variance at each time step
expected_logv = (dist * X_grid).sum(axis=1)
expected_var = np.exp(expected_logv)

# Initialize price nodes
price_nodes = np.zeros((N+1, N+1))
price_nodes[0,0] = S0

for t in range(1, N+1):
    vol_t = math.sqrt(max(expected_var[t-1], eps))
    u = math.exp(vol_t * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r-q)*dt) - d) / (u - d)
    p = float(np.clip(p, 0.0, 1.0))
    
    for j in range(t+1):
        if j == 0:
            price_nodes[t,j] = price_nodes[t-1,0] * d
        elif j == t:
            price_nodes[t,j] = price_nodes[t-1,j-1] * u
        else:
            price_nodes[t,j] = p * price_nodes[t-1,j-1] * u + (1-p) * price_nodes[t-1,j] * d

# Option values at maturity
V_next = np.zeros(N+1)
for j in range(N+1):
    V_next[j] = max(price_nodes[N,j] - K, 0.0)

# Backward induction
for t in range(N-1, -1, -1):
    V_curr = np.zeros(t+1)
    for j in range(t+1):
        vol_t = math.sqrt(max(expected_var[t], eps))
        u = math.exp(vol_t * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp((r-q)*dt) - d) / (u - d)
        p = float(np.clip(p, 0.0, 1.0))
        V_curr[j] = math.exp(-r*dt) * (p*V_next[j+1] + (1-p)*V_next[j])
    V_next = V_curr

binomial_price = V_next[0]
print(f"Binomial Tree Price (Log-OU): ${binomial_price:.4f}")

# Estimate correlation between price returns and volatility changes
log_returns = np.log(prices / prices.shift(1)).dropna()
vol_changes = volatility_historical.diff().dropna()
aligned_data = pd.concat([log_returns, vol_changes], axis=1).dropna()
aligned_data.columns = ['returns', 'vol_changes']
rho = aligned_data.corr().iloc[0, 1]
print(f"Estimated correlation between price and volatility: {rho:.4f}")

# --- Monte Carlo simulation using safe rho ---
num_simulations = 50000
if num_simulations % 2 == 1:
    num_simulations += 1

dt = T / N if N > 0 else 0.0
np.random.seed(42)

num_half = num_simulations // 2
Z_half = np.random.normal(size=(num_half, 2, N))
Z_full = np.concatenate([Z_half, -Z_half], axis=0)  # antithetic

log_v = np.full(num_simulations, np.log(max(v0, 1e-12)))
S = np.full(num_simulations, S0)

for j in range(N):
    z = Z_full[:, :, j]
    vol_shock, price_shock = z[:, 0], z[:, 1]

    log_v = mu_ou + (log_v - mu_ou) * np.exp(-k*dt) + sigma_ou_const * vol_shock
    variance = np.exp(np.maximum(log_v, np.log(1e-12)))
    sigma = np.sqrt(variance)

    S *= np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * price_shock)

payoffs = np.maximum(S - K, 0.0)
mc_price = np.exp(-r * T) * payoffs.mean()
mc_std = np.exp(-r * T) * payoffs.std(ddof=1) / np.sqrt(num_simulations)

print(f"Monte Carlo Price: ${mc_price:.4f} (±${1.96*mc_std:.4f} 95% CI)")


print(f"Monte Carlo Price (antithetic, rho={rho:.3f}): ${mc_price:.4f} (±${mc_std*1.96:.4f} 95% CI)")
print(f"\nComparison for {ticker} {exp_date} Call Option with Strike {K}:")
print(f"Market Price: ${market_price:.2f}")
print(f"Binomial Tree Price: ${binomial_price:.2f}")
print(f"Monte Carlo Price: ${mc_price:.2f} (±${mc_std*1.96:.2f} with 95% CI)")

if market_price > 0:
    binomial_diff_pct = (binomial_price - market_price) / market_price * 100
    mc_diff_pct = (mc_price - market_price) / market_price * 100
    print(f"Percentage Difference (Binomial): {binomial_diff_pct:.2f}%")
    print(f"Percentage Difference (MC): {mc_diff_pct:.2f}%")
    