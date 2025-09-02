import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats, optimize

# 1. Εισαγωγή δεδομένων από τον χρήστη
try:
    ticker = input("Εισάγετε το ticker symbol της μετοχής (π.χ. TSLA, AAPL): ").strip().upper()
    if not ticker:
        raise ValueError("Το ticker δεν μπορεί να είναι κενό.")
    
    N = int(input("Εισάγετε τον αριθμό των χρονικών διαστημάτων (N) [π.χ. 252]: "))
    if N <= 0:
        raise ValueError("Ο αριθμός N πρέπει να είναι θετικός.")
    
    end_date = datetime(2025, 9, 2).strftime('%Y-%m-%d')  # Σημερινή ημερομηνία
    start_date = (datetime(2025, 9, 2) - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 χρόνια πίσω
    print(f"Start date: {start_date}, End date: {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"Δεν βρέθηκαν δεδομένα για το ticker {ticker}. Ελέγξτε το όνομα.")
    
    S0 = df['Close'].iloc[-1].item()
    prices = df['Close']
    print(f"\nInitial stock price για {ticker}: {S0:.2f}")
except Exception as e:
    print(f"Error fetching data: {e}")
    S0 = 150.0  # Αν είναι κενό βάζουμε μια συγκεκριμένη τιμή
    print(f"\nUsing fallback stock price: {S0}")

# Compute historical volatility for MLE
log_returns = np.log(prices / prices.shift(1)).dropna()
volatility_historical = log_returns.rolling(window=252).std() * np.sqrt(252)  # Increased to 252 days
variance_historical = volatility_historical ** 2
log_variance = np.log(variance_historical.ffill().dropna())  # Log of variance
print("Sample of Historical Volatility (sigma):\n")
print(volatility_historical.head())
print("Sample of Log Variance (log(v_t)):\n")
print(log_variance.head())
print("\n")

# Define likelihood functions for MLE
def mu(log_v_t, dt, k, theta):
    ekt = np.exp(-k * dt)
    return log_v_t * ekt + np.log(theta) * (1 - ekt)

def std(dt, k, x):
    e2kt = np.exp(-2 * k * dt)
    std_val = x * np.sqrt((1 - e2kt) / (2 * k))
    return np.where(std_val > 0, std_val, 1e-10)  # Avoid zero or negative std

def log_likelihood(params, log_vol, dt):
    k, theta, x = params
    if k <= 0 or theta <= 0 or x <= 0:
        return np.inf
    log_v_t = log_vol[:-1]
    log_v_dt = log_vol[1:]
    if len(log_v_t) != len(log_v_dt) or len(log_v_t) < 2:
        return np.inf
    mu_OU = mu(log_v_t, dt, k, theta)
    sigma_OU = std(dt, k, x)
    ll = np.sum(stats.norm.logpdf(log_v_dt, loc=mu_OU, scale=sigma_OU))
    return -ll if np.isfinite(ll) else np.inf

# Calibrate Log-OU parameters with MLE
dt = 1 / 252  # Daily time step for calibration
initial_guesses = [[1.0, 0.04, 0.1], [0.5, 0.05, 0.05], [1.5, 0.03, 0.15]]  # Multiple initial guesses
bounds = [(0.01, 5.0), (0.01, 0.2), (0.01, 0.2)]  # Adjusted theta bound to 0.2
best_ll = np.inf
best_params = [2.0, 0.04, 0.1]  # Fallback
for guess in initial_guesses:
    result = optimize.minimize(log_likelihood, guess, args=(log_variance, dt), bounds=bounds, method='L-BFGS-B')
    if result.success and result.fun < best_ll:
        best_ll = result.fun
        best_params = result.x
k, theta, x = best_params
v0 = theta  # Initial variance set to long-term mean
print(f"{ticker} Calibrated Parameters (via MLE):\n")
print(f"k = {k:.4f}, theta = {theta:.4f}, x = {x:.4f}\n")
print(f"Optimization success: {result.success}, Message: {result.message}\n")
if not result.success:
    print("Calibration failed. Using fallback parameters.\n")
    k, theta, x, v0 = 2.0, 0.04, 0.1, 0.04

# Παράμετροι για το δέντρο
T = 1.0  # Χρόνος λήξης
dt = T / N  # Χρονικά διαστήματα βασισμένα στο N
r = 0.03  # Ρίσκο ελεύθερο επιτόκιο
K = S0 * 1.1  # Out of the money strike price

# log-OU υπολογισμός
np.random.seed(42)  # Οι τυχαίοι αριθμοί είναι ίδιοι
log_v = np.zeros(N + 1)
log_v[0] = np.log(v0)
for i in range(1, N + 1):
    log_v[i] = log_v[i-1] + k * (np.log(theta) - log_v[i-1]) * dt + x * np.sqrt(dt) * np.random.normal()
volatility = np.sqrt(np.exp(log_v))  # Μεταβλητότητα
print(f"\nVolatility range for {ticker}: min = {np.min(volatility):.4f}, max = {np.max(volatility):.4f}, mean = {np.mean(volatility):.4f}")

# Building the binomial tree
tree = np.zeros((N + 1, N + 1))
tree[0, 0] = S0
for i in range(N):
    sigma = volatility[i]  # Θέτω από το log-OU
    up = np.exp(sigma * np.sqrt(dt))  # Πάνω μέρος του δέντρου
    down = 1 / up  # Κάτω μέρος του δέντρου
    prob = (np.exp(r * dt) - down) / (up - down)  # Risk-neutral πιθανότητα
    for j in range(i + 1):
        if j == 0:
            tree[i + 1, j] = tree[i, j] * down
        else:
            tree[i + 1, j] = tree[i, j - 1] * up  # Δημιουργία κόμβων στον δέντρο

# Υπολογισμός της τιμής του option
option_price = np.zeros((N + 1, N + 1))
for j in range(N + 1):
    option_price[N, j] = max(0, tree[N, j] - K)

for i in range(N - 1, -1, -1):  # Discounting the Prices
    sigma = volatility[i]
    up = np.exp(sigma * np.sqrt(dt))
    down = 1 / up
    prob = (np.exp(r * dt) - down) / (up - down)
    for j in range(i + 1):
        option_price[i, j] = np.exp(-r * dt) * (prob * option_price[i + 1, j + 1] + (1 - prob) * option_price[i + 1, j])

# 6. Εκτύπωση αποτελέσματος
print(f"\nCall Option Price for {ticker}: {option_price[0, 0]:.2f}")

# 7. Οπτικοποίηση της προσομοιωμένης μεταβλητότητας
plt.plot(volatility)
plt.title(f'Log-Ornstein-Uhlenbeck Volatility for {ticker}')
plt.xlabel('Time Steps')
plt.ylabel('Volatility (σ)')
plt.savefig(f'volatility_plot_{ticker}.png')  # Save the plot with ticker name
plt.show()