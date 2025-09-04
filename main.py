import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Fetch user inputs
ticker = input("Enter stock ticker (e.g., AAPL, NVDA): ").strip().upper()
N = int(input("Number of time steps for binomial tree (e.g., 100): "))

# Download historical data
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
start_date = (pd.Timestamp.now() - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    raise ValueError(f"No data found for {ticker}")
S0 = float(data['Close'].iloc[-1])
prices = data['Close']

print(f"Current {ticker} price: ${S0:.2f}")

# Calculate log returns and historical volatility
log_returns = np.log(prices / prices.shift(1)).dropna()
volatility_historical = log_returns.rolling(window=252).std() * np.sqrt(252)
variance_historical = volatility_historical.ffill().dropna()
log_variance = np.log(variance_historical)

# MLE Calibration for Log-OU process
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
v0 = variance_historical.iloc[-1]  # Use the most recent historical variance

print(f"Calibrated Log-OU parameters: k={k:.4f}, theta={theta:.4f}, x={x:.4f}")
print(f"Calibrated annual volatility: {np.sqrt(theta)*100:.2f}%")
print(f"Current annual volatility: {np.sqrt(v0.values[0])*100:.2f}%")

# Fetch option data for the stock
stock = yf.Ticker(ticker)
exp_dates = stock.options
if not exp_dates:
    raise ValueError(f"No option data for {ticker}")
    
# Let user select expiration date
print("\nAvailable expiration dates:")
for i, date in enumerate(exp_dates):
    print(f"{i+1}. {date}")
exp_choice = int(input("Select expiration date by number: ")) - 1
exp_date = exp_dates[exp_choice]

# Calculate time to expiration
T = (pd.Timestamp(exp_date) - pd.Timestamp.now()).days / 365
print(f"Time to expiration: {T:.2f} years")

# Set risk-free rate and dividend yield
try:
    treasury = yf.Ticker("^TNX")
    treasury_data = treasury.history(period="1d")
    r = treasury_data['Close'].iloc[-1] / 100
    print(f"Current 10-year Treasury yield: {r:.4f}")
except:
    r = 0.02  # Fallback rate
    print(f"Using fallback risk-free rate: {r:.4f}")

q = stock.info.get('dividendYield', 0)  # Use actual dividend yield if available
print(f"Using risk-free rate: {r:.4f}, dividend yield: {q:.4f}")

# Get options for selected expiration
options = stock.option_chain(exp_date)
calls = options.calls

# Filter and sort options
valid_calls = calls[((calls['bid'] > 0) | (calls['ask'] > 0))].copy()
if 'midPrice' not in valid_calls.columns:
    valid_calls['midPrice'] = (valid_calls['bid'] + valid_calls['ask']) / 2

valid_calls['moneyness'] = valid_calls['strike'] / S0
valid_calls['distance'] = abs(valid_calls['moneyness'] - 1)

# Sort by moneyness distance
if 'volume' in valid_calls.columns and 'openInterest' in valid_calls.columns:
    valid_calls = valid_calls.sort_values(['distance', 'volume', 'openInterest'], 
                                         ascending=[True, False, False])
else:
    valid_calls = valid_calls.sort_values('distance', ascending=True)

# Display options
valid_calls_display = valid_calls.reset_index(drop=True)
print(f"\nAvailable call options (sorted by distance from current price ${S0:.2f}):")
display_cols = ['strike', 'bid', 'ask', 'midPrice', 'moneyness']
if 'impliedVolatility' in valid_calls_display.columns:
    display_cols.append('impliedVolatility')
    
display_cols = [col for col in display_cols if col in valid_calls_display.columns]
valid_calls_display['Index'] = valid_calls_display.index
display_cols = ['Index'] + display_cols

print(valid_calls_display[display_cols].head(10))

# Select strike price
print("\nOptions:")
print("1. Select from the list above")
print("2. Enter a custom strike price")
choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    num_options = min(10, len(valid_calls_display))
    strike_choice = int(input(f"Enter the index of the strike price you want to use (0-{num_options-1}): "))
    if strike_choice < 0 or strike_choice >= num_options:
        raise ValueError(f"Invalid index. Please enter a number between 0 and {num_options-1}")
    call_option = valid_calls_display.iloc[strike_choice]
    K = call_option['strike']
    market_price = call_option['midPrice']
else:
    K = float(input("Enter the custom strike price: "))
    # Find the closest option to get market price
    closest_idx = (valid_calls['strike'] - K).abs().idxmin()
    if abs(valid_calls.loc[closest_idx, 'strike'] - K) / K < 0.05:  # Within 5%
        market_price = valid_calls.loc[closest_idx, 'midPrice']
        print(f"Using market price ${market_price:.2f} from similar strike")
    else:
        # Estimate price using Black-Scholes with historical volatility
        from scipy.stats import norm
        d1 = (np.log(S0/K) + (r - q + 0.5*theta)*T) / (np.sqrt(theta)*np.sqrt(T))
        d2 = d1 - np.sqrt(theta)*np.sqrt(T)
        market_price = S0 * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        print(f"Estimated market price using Black-Scholes: ${market_price:.2f}")

# Always use historically calibrated volatility
print("Using historically calibrated volatility")

print(f"\nPricing {ticker} {exp_date} Call Option with Strike {K}:")

# Binomial tree with stochastic volatility (Log-OU)
print("Building binomial tree with Log-OU stochastic volatility...")

dt = T / N

# We'll create a 2D tree for both price and variance
price_tree = np.zeros((N+1, N+1))
variance_tree = np.zeros((N+1, N+1))

# Initialize
price_tree[0, 0] = S0
variance_tree[0, 0] = v0

# Build the trees
for i in range(1, N+1):
    for j in range(i+1):
        # Update variance using Log-OU process
        if j == 0:
            # Down movement for variance
            prev_var = variance_tree[i-1, j]
            log_var = np.log(max(prev_var, 1e-10))
            log_var = log_var + k * (np.log(theta) - log_var) * dt - x * np.sqrt(dt)
            variance_tree[i, j] = max(np.exp(log_var), 1e-10)
        elif j == i:
            # Up movement for variance
            prev_var = variance_tree[i-1, j-1]
            log_var = np.log(max(prev_var, 1e-10))
            log_var = log_var + k * (np.log(theta) - log_var) * dt + x * np.sqrt(dt)
            variance_tree[i, j] = max(np.exp(log_var), 1e-10)
        else:
            # Average of up and down movements for variance (recombining approximation)
            prev_var_down = variance_tree[i-1, j]
            prev_var_up = variance_tree[i-1, j-1]
            log_var_down = np.log(max(prev_var_down, 1e-10))
            log_var_up = np.log(max(prev_var_up, 1e-10))
            
            # Average the log variances
            avg_log_var = (log_var_down + log_var_up) / 2
            avg_log_var = avg_log_var + k * (np.log(theta) - avg_log_var) * dt
            variance_tree[i, j] = max(np.exp(avg_log_var), 1e-10)
        
        # Update price based on current volatility
        volatility = np.sqrt(variance_tree[i, j])
        u = np.exp(volatility * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Calculate price
        if j == 0:
            price_tree[i, j] = price_tree[i-1, j] * d
        elif j == i:
            price_tree[i, j] = price_tree[i-1, j-1] * u
        else:
            price_tree[i, j] = p * price_tree[i-1, j-1] * u + (1-p) * price_tree[i-1, j] * d

# Calculate option value at expiration
option_value = np.zeros((N+1, N+1))
for j in range(N+1):
    option_value[N, j] = max(price_tree[N, j] - K, 0)

# Backward induction for option pricing
for i in range(N-1, -1, -1):
    for j in range(i+1):
        volatility = np.sqrt(variance_tree[i, j])
        u = np.exp(volatility * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Expected value of option
        option_value[i, j] = np.exp(-r * dt) * (p * option_value[i+1, j+1] + (1-p) * option_value[i+1, j])

binomial_price = option_value[0, 0]

# Monte Carlo simulation for comparison
print("Running Monte Carlo simulation for comparison...")
num_simulations = 50000

np.random.seed(42)
# Use correlated shocks for more realistic simulation
shocks = np.random.multivariate_normal(
    mean=[0, 0], 
    cov=[[1, 0.2], [0.2, 1]],  # Add some correlation between price and volatility shocks
    size=(num_simulations, N)
)
vol_shocks = shocks[:, :, 0]
price_shocks = shocks[:, :, 1]

mc_payoffs = np.zeros(num_simulations)
for i in range(num_simulations):
    v0_value = v0.iloc[0] if isinstance(v0, pd.Series) else v0
    log_v = np.log(max(v0_value, 1e-10))
    S = S0
    
    for j in range(N):
        # Update volatility process
        log_v = log_v + k * (np.log(theta) - log_v) * dt + x * np.sqrt(dt) * vol_shocks[i, j]
        variance = max(np.exp(log_v), 1e-10)
        sigma = np.sqrt(variance)
        
        # Update stock price
        S = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * price_shocks[i, j])
    
    mc_payoffs[i] = max(S - K, 0)

mc_price = np.exp(-r * T) * np.mean(mc_payoffs)
mc_std = np.exp(-r * T) * np.std(mc_payoffs) / np.sqrt(num_simulations)

# Results
print(f"\nComparison for {ticker} {exp_date} Call Option with Strike {K}:")
print(f"Market Price: ${market_price:.2f}")
print(f"Binomial Tree Price: ${binomial_price:.2f}")
print(f"Monte Carlo Price: ${mc_price:.2f} (±${mc_std*1.96:.2f} with 95% CI)")

if market_price > 0:
    binomial_diff_pct = (binomial_price - market_price) / market_price * 100
    mc_diff_pct = (mc_price - market_price) / market_price * 100
    print(f"Percentage Difference (Binomial): {binomial_diff_pct:.2f}%")
    print(f"Percentage Difference (MC): {mc_diff_pct:.2f}%")

# Additional analysis
print("\nAdditional Analysis:")
print(f"Option Moneyness: {K/S0:.2%} (Strike/Current Price)")
intrinsic_value = max(S0 - K, 0)
print(f"Intrinsic Value: ${intrinsic_value:.2f}")
print(f"Time Value: ${market_price - intrinsic_value:.2f}")