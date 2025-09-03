import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Fetch user inputs
ticker = input("Enter stock ticker (e.g., AAPL, NVDA): ").strip().upper()
N = int(input("Number of time steps (e.g., 252): "))

# Download historical data
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
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
k_ou, theta_ou, x_ou = result.x
v0_ou = theta_ou

print(f"Calibrated Log-OU parameters: k={k_ou:.4f}, theta={theta_ou:.4f}, x={x_ou:.4f}")
print(f"Calibrated annual volatility: {np.sqrt(theta_ou)*100:.2f}%")

# Try Heston model calibration if enough data
k_heston, theta_heston, sigma_heston, rho_heston, v0_heston = None, None, None, None, None
if len(log_returns) > 100:
    def heston_log_likelihood(params, returns, dt=1/252):
        k_heston, theta_heston, sigma_heston, rho_heston, v0_heston = params
        if k_heston <= 0 or theta_heston <= 0 or sigma_heston <= 0 or abs(rho_heston) >= 1 or v0_heston <= 0:
            return np.inf
        
        # Use numpy arrays for calculations
        returns = np.array(returns)
        n = len(returns)
        ll = -n/2 * np.log(2*np.pi) - n/2 * np.log(theta_heston*dt) - 1/(2*theta_heston*dt) * np.sum(returns**2)
        return -ll

    heston_bounds = [(1e-3, 10), (1e-3, 0.5), (1e-3, 0.5), (-0.99, 0.99), (1e-3, 0.5)]
    try:
        heston_result = differential_evolution(heston_log_likelihood, heston_bounds, 
                                              args=(log_returns,), maxiter=500, seed=42)
        k_heston, theta_heston, sigma_heston, rho_heston, v0_heston = heston_result.x
        print(f"Heston calibrated parameters: k={k_heston:.4f}, theta={theta_heston:.4f}, "
              f"sigma={sigma_heston:.4f}, rho={rho_heston:.4f}, v0={v0_heston:.4f}")
        print(f"Heston calibrated annual volatility: {np.sqrt(theta_heston)*100:.2f}%")
    except Exception as e:
        print(f"Heston calibration failed: {e}")

# Model selection
use_heston = False
if k_heston is not None:
    print("\nModel calibration results:")
    print(f"Log-OU: k={k_ou:.4f}, theta={theta_ou:.4f}, xi={x_ou:.4f}")
    print(f"Heston: k={k_heston:.4f}, theta={theta_heston:.4f}, sigma={sigma_heston:.4f}, rho={rho_heston:.4f}")
    
    use_heston_input = input("Use Heston model instead? (y/n): ").strip().lower()
    if use_heston_input == 'y':
        use_heston = True
        k, theta, sigma, rho, v0 = k_heston, theta_heston, sigma_heston, rho_heston, v0_heston
        print("Using Heston model parameters")
    else:
        k, theta, x, v0 = k_ou, theta_ou, x_ou, v0_ou
        print("Using Log-OU model parameters")
else:
    k, theta, x, v0 = k_ou, theta_ou, x_ou, v0_ou
    print("Using Log-OU model (Heston calibration failed)")

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

# Calculate time to expiration for dynamic moneyness range
T = (pd.Timestamp(exp_date) - pd.Timestamp.now()).days / 365
if T > 1:  # If more than 1 year to expiration
    moneyness_range = 0.4  # 40% range
else:
    moneyness_range = 0.2  # 20% range

# Get options for selected expiration
options = stock.option_chain(exp_date)
calls = options.calls

# Create a comprehensive filtering approach
filter_conditions = [
    (calls['bid'] > 0) | (calls['ask'] > 0),  # At least one is positive
    calls['volume'] > 0 if 'volume' in calls.columns else True,  # Has volume if available
    calls['openInterest'] > 0 if 'openInterest' in calls.columns else True  # Has open interest if available
]

# Combine filters
combined_filter = pd.Series(True, index=calls.index)
for condition in filter_conditions:
    combined_filter &= condition

valid_calls = calls[combined_filter].copy()

if valid_calls.empty:
    # If no options meet criteria, use all calls with positive bid/ask
    valid_calls = calls[((calls['bid'] > 0) | (calls['ask'] > 0))].copy()

# Calculate moneyness and prioritize near-the-money options
valid_calls['moneyness'] = valid_calls['strike'] / S0
valid_calls['distance'] = abs(valid_calls['moneyness'] - 1)

# Sort by moneyness distance and liquidity
if 'volume' in valid_calls.columns and 'openInterest' in valid_calls.columns:
    valid_calls = valid_calls.sort_values(['distance', 'volume', 'openInterest'], 
                                         ascending=[True, False, False])
else:
    valid_calls = valid_calls.sort_values('distance', ascending=True)

# Calculate midPrice if not available
if 'midPrice' not in valid_calls.columns:
    valid_calls['midPrice'] = (valid_calls['bid'] + valid_calls['ask']) / 2

# Reset index for cleaner display
valid_calls_display = valid_calls.reset_index(drop=True)

# Show available options with moneyness information
print(f"\nAvailable call options (sorted by distance from current price ${S0:.2f}):")
display_cols = ['strike', 'bid', 'ask', 'midPrice', 'moneyness']
if 'impliedVolatility' in valid_calls_display.columns:
    display_cols.append('impliedVolatility')
if 'volume' in valid_calls_display.columns:
    display_cols.append('volume')
if 'openInterest' in valid_calls_display.columns:
    display_cols.append('openInterest')
    
# Filter to only existing columns
display_cols = [col for col in display_cols if col in valid_calls_display.columns]

# Add index column for selection
valid_calls_display['Index'] = valid_calls_display.index
display_cols = ['Index'] + display_cols

print(valid_calls_display[display_cols].head(15))

# Let user select from the sorted list or enter a custom strike
print("\nOptions:")
print("1. Select from the list above")
print("2. Enter a custom strike price")
choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    num_options = min(15, len(valid_calls_display))
    strike_choice = int(input(f"Enter the index of the strike price you want to use (0-{num_options-1}): "))
    if strike_choice < 0 or strike_choice >= num_options:
        raise ValueError(f"Invalid index. Please enter a number between 0 and {num_options-1}")
    call_option = valid_calls_display.iloc[strike_choice]
    K = call_option['strike']
    market_price = call_option['midPrice']
else:
    K = float(input("Enter the custom strike price: "))
    # Find the closest option to get market price, or use Black-Scholes approximation
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

# Use implied volatility if available and reasonable, otherwise use calibrated volatility
if 'impliedVolatility' in call_option and not pd.isna(call_option['impliedVolatility']) and call_option['impliedVolatility'] > 0.05:
    implied_vol = call_option['impliedVolatility']
    print(f"Using implied volatility: {implied_vol:.4f}")
    # Adjust calibration to match implied volatility
    theta = implied_vol**2  # Set long-term variance to implied variance
    v0 = theta
else:
    print("Using historically calibrated volatility")
    # Use the calibrated parameters as is

print(f"Time to expiration: {T:.2f} years")

# Set risk-free rate and dividend yield
# Get current risk-free rate from 10-year Treasury yield
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

# Different simulation methods based on model selection
dt = T / N
num_simulations = 100000

if use_heston:
    # Heston model simulation
    print("Using Heston model simulation")
    
    # Generate correlated Brownian motions
    np.random.seed(42)
    Z1 = np.random.normal(0, 1, (num_simulations, N))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (num_simulations, N))
    
    # Initialize arrays
    S_paths = np.zeros((num_simulations, N+1))
    v_paths = np.zeros((num_simulations, N+1))
    
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    
    # Heston simulation
    for t in range(1, N+1):
        # Ensure variance doesn't go negative using full truncation
        v_prev = np.maximum(v_paths[:, t-1], 0)
        sqrt_v_prev = np.sqrt(v_prev)
        
        # Update variance
        v_paths[:, t] = v_prev + k * (theta - v_prev) * dt + sigma * sqrt_v_prev * np.sqrt(dt) * Z2[:, t-1]
        
        # Update stock price
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - q - 0.5 * v_prev) * dt + sqrt_v_prev * np.sqrt(dt) * Z1[:, t-1])
    
    # Calculate payoffs
    mc_payoffs = np.maximum(S_paths[:, -1] - K, 0)
    mc_price = np.exp(-r * T) * np.mean(mc_payoffs)
    mc_std = np.exp(-r * T) * np.std(mc_payoffs) / np.sqrt(num_simulations)
    
    # Binomial tree not implemented for Heston
    binomial_price = np.nan
    
else:
    # Log-OU model simulation
    print("Using Log-OU model simulation")
    
    # Generate volatility path with bounds to prevent numerical issues
    log_v = np.log(max(v0, 1e-10))  # Ensure v0 is positive
    volatility_path = []
    for i in range(N):
        log_v = log_v + k * (np.log(theta) - log_v) * dt + x * np.sqrt(dt) * np.random.normal()
        # Ensure variance doesn't go to zero or negative
        variance = max(np.exp(log_v), 1e-10)
        volatility_path.append(np.sqrt(variance))

    # Build binomial tree with improved numerical stability
    tree = np.zeros((N+1, N+1))
    tree[0, 0] = S0

    for i in range(1, N+1):
        sigma = volatility_path[i-1]
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        # Ensure probability is between 0 and 1
        p = max(0, min(1, (np.exp((r - q) * dt) - d) / (u - d)))
        
        for j in range(i+1):
            if j == 0:
                tree[i, j] = tree[i-1, j] * d
            else:
                tree[i, j] = tree[i-1, j-1] * u

    # Option pricing via binomial tree with error handling
    try:
        option_value = np.maximum(tree[-1, :] - K, 0)
        for i in range(N-1, -1, -1):
            sigma = volatility_path[i]
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = max(0, min(1, (np.exp((r - q) * dt) - d) / (u - d)))
            
            for j in range(i+1):
                option_value[j] = np.exp(-r * dt) * (p * option_value[j+1] + (1-p) * option_value[j])
        
        binomial_price = option_value[0]
    except Exception as e:
        print(f"Error in binomial tree calculation: {e}. Using Monte Carlo only.")
        binomial_price = np.nan

    # Monte Carlo simulation for Log-OU
    np.random.seed(42)
    vol_shocks = np.random.normal(0, 1, (num_simulations, N))
    np.random.seed(43)
    price_shocks = np.random.normal(0, 1, (num_simulations, N))

    mc_payoffs = np.zeros(num_simulations)
    for i in range(num_simulations):
        log_v = np.log(max(v0, 1e-10))  # Ensure v0 is positive
        S = S0
        
        for j in range(N):
            # Update volatility process
            log_v = log_v + k * (np.log(theta) - log_v) * dt + x * np.sqrt(dt) * vol_shocks[i, j]
            # Ensure variance doesn't go to zero or negative
            variance = max(np.exp(log_v), 1e-10)
            sigma = np.sqrt(variance)
            
            # Update stock price
            S = S * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * price_shocks[i, j])
        
        mc_payoffs[i] = max(S - K, 0)

    mc_price = np.exp(-r * T) * np.mean(mc_payoffs)
    mc_std = np.exp(-r * T) * np.std(mc_payoffs) / np.sqrt(num_simulations)

# Calculate Greeks using finite differences
def calculate_greeks(S0, K, T, r, q, params, pricing_function, delta=0.01):
    if use_heston:
        k, theta, sigma, rho, v0 = params
    else:
        k, theta, x, v0 = params
    
    # Delta
    price_base = pricing_function(S0, K, T, r, q, params)
    price_up = pricing_function(S0*(1+delta), K, T, r, q, params)
    price_down = pricing_function(S0*(1-delta), K, T, r, q, params)
    delta_val = (price_up - price_down) / (2 * S0 * delta)
    
    # Gamma
    gamma_val = (price_up - 2*price_base + price_down) / (S0 * delta)**2
    
    # Vega (change in volatility)
    if use_heston:
        # For Heston, bump initial volatility
        v0_bumped = v0 * (1 + delta)
        params_bumped = (k, theta, sigma, rho, v0_bumped)
    else:
        # For Log-OU, bump long-term variance (theta)
        theta_bumped = theta * (1 + delta)
        params_bumped = (k, theta_bumped, x, v0)
        
    price_vega = pricing_function(S0, K, T, r, q, params_bumped)
    vega_val = (price_vega - price_base) / (delta * 100)  # Vega per 1% change in vol
    
    # Theta (time decay)
    time_shift = max(T * 0.01, 1/365)  # At least 1 day
    T_shifted = max(T - time_shift, 1/365)  # Ensure positive time
    price_theta = pricing_function(S0, K, T_shifted, r, q, params)
    theta_val = (price_theta - price_base) / time_shift  # Theta per year
    
    # Convert to daily theta (divide by 365)
    theta_daily = theta_val / 365
    
    # Rho (interest rate sensitivity)
    price_rho = pricing_function(S0, K, T, r*(1+delta), q, params)
    rho_val = (price_rho - price_base) / (r * delta)  # Rho per 1% change in rate
    
    return delta_val, gamma_val, vega_val, theta_daily, rho_val

# Define pricing function for Monte Carlo
def mc_pricing_function(S, K, T, r, q, params):
    if use_heston:
        k, theta, sigma, rho, v0 = params
        # Heston simulation with fewer paths for Greek calculation
        num_simulations_greek = 10000
        
        # Generate correlated Brownian motions
        np.random.seed(42)
        Z1 = np.random.normal(0, 1, (num_simulations_greek, N))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (num_simulations_greek, N))
        
        # Initialize arrays
        S_paths = np.zeros((num_simulations_greek, N+1))
        v_paths = np.zeros((num_simulations_greek, N+1))
        
        S_paths[:, 0] = S
        v_paths[:, 0] = v0
        
        dt = T / N
        
        # Heston simulation
        for t in range(1, N+1):
            # Ensure variance doesn't go negative using full truncation
            v_prev = np.maximum(v_paths[:, t-1], 0)
            sqrt_v_prev = np.sqrt(v_prev)
            
            # Update variance
            v_paths[:, t] = v_prev + k * (theta - v_prev) * dt + sigma * sqrt_v_prev * np.sqrt(dt) * Z2[:, t-1]
            
            # Update stock price
            S_paths[:, t] = S_paths[:, t-1] * np.exp((r - q - 0.5 * v_prev) * dt + sqrt_v_prev * np.sqrt(dt) * Z1[:, t-1])
        
        # Calculate payoffs
        payoffs = np.maximum(S_paths[:, -1] - K, 0)
        return np.exp(-r * T) * np.mean(payoffs)
        
    else:
        k, theta, x, v0 = params
        # Log-OU simulation with fewer paths for Greek calculation
        num_simulations_greek = 10000
        
        np.random.seed(42)
        vol_shocks = np.random.normal(0, 1, (num_simulations_greek, N))
        np.random.seed(43)
        price_shocks = np.random.normal(0, 1, (num_simulations_greek, N))
        
        payoffs = np.zeros(num_simulations_greek)
        dt = T / N
        
        for i in range(num_simulations_greek):
            log_v = np.log(max(v0, 1e-10))
            St = S
            
            for j in range(N):
                # Update volatility process
                log_v = log_v + k * (np.log(theta) - log_v) * dt + x * np.sqrt(dt) * vol_shocks[i, j]
                variance = max(np.exp(log_v), 1e-10)
                sigma = np.sqrt(variance)
                
                # Update stock price
                St = St * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * price_shocks[i, j])
            
            payoffs[i] = max(St - K, 0)
        
        return np.exp(-r * T) * np.mean(payoffs)

# Calculate Greeks for Monte Carlo
if use_heston:
    mc_greeks = calculate_greeks(S0, K, T, r, q, (k, theta, sigma, rho, v0), mc_pricing_function)
else:
    mc_greeks = calculate_greeks(S0, K, T, r, q, (k, theta, x, v0), mc_pricing_function)

# Compare with market price
print(f"\nComparison for {ticker} {exp_date} Call Option with Strike {K}:")
print(f"Market Price: ${market_price:.2f}")
if not np.isnan(binomial_price):
    print(f"Binomial Tree Price: ${binomial_price:.2f}")
    print(f"Difference from Market (Binomial): ${binomial_price - market_price:.2f}")
print(f"Monte Carlo Price: ${mc_price:.2f} (±${mc_std*1.96:.2f} with 95% CI)")
print(f"Difference from Market (MC): ${mc_price - market_price:.2f}")

# Calculate percentage differences
if market_price > 0:
    if not np.isnan(binomial_price):
        binomial_diff_pct = (binomial_price - market_price) / market_price * 100
        print(f"Percentage Difference (Binomial): {binomial_diff_pct:.2f}%")
    mc_diff_pct = (mc_price - market_price) / market_price * 100
    print(f"Percentage Difference (MC): {mc_diff_pct:.2f}%")
else:
    print("Market price is zero, cannot calculate percentage difference")

# Additional analysis
print("\nAdditional Analysis:")
print(f"Option Moneyness: {K/S0:.2%} (Strike/Current Price)")
intrinsic_value = max(S0 - K, 0)
print(f"Intrinsic Value: ${intrinsic_value:.2f}")
print(f"Time Value: ${market_price - intrinsic_value:.2f}")

# Risk Measures (Greeks)
print("\nRisk Measures (Greeks) - Monte Carlo:")
print(f"Delta: {mc_greeks[0]:.4f} (price change per $1 change in underlying)")
print(f"Gamma: {mc_greeks[1]:.6f} (delta change per $1 change in underlying)")
print(f"Vega: {mc_greeks[2]:.4f} (price change per 1% change in volatility)")
print(f"Theta: {mc_greeks[3]:.4f} (price change per day)")
print(f"Rho: {mc_greeks[4]:.4f} (price change per 1% change in interest rate)")