import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. Δεδομενα
try:
    end_date = datetime(2025, 9, 2).strftime('%Y-%m-%d')  # Σημερινη ημερα
    start_date = (datetime(2025, 9, 2) - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 xρονια πισω
    df = yf.download("TSLA", start=start_date, end=end_date, auto_adjust=True)  # αυτοματα βαζει ημερομηνιες
    
    if df.empty:
        raise ValueError("No data retrieved from yfinance. Check ticker or date range.")
    
    # Ελεγχουμε αν το dataframe ειναι κενο
    S0 = df['Close'].iloc[-1].item()  
    print(f"Initial stock price (S0): {S0:.2f}")
except Exception as e:
    print(f"Error fetching data: {e}")
    S0 = 150.0  # Αν ειναι κενο βαζουμε μια συγκεκριμενη τιμη
    print(f"Using fallback stock price: {S0}") 

# Παραμετροι για το δεντρο 
T = 1.0 # Χρονος Ληξης
N = 252
dt = T/N # Χρονικα διαστηματα 
r = 0.05 # ρισκ φρεε
K = S0 * 1.1 # 1.1 ωστε να ειμαιστε out of the money και οχι at the money

# Παραμετροι για το Log-ou
k = 2.0 # ποσο γρηγορα επιστρεφη στο μεσο ορο μετα την αποκλιση
theta = 0.04 # εση τιμη της διακυμανσης
x = 0.1 # volatility of volatility
v0 = 0.04 # αρχικη τιμη της διακυμανσης


# log-OU υπολογισμος
np.random.seed(42) # οι τυχαιοι αριθμοι ειναι ιδιοι
log_v = np.zeros(N+1)
log_v[0] = np.log(v0)
for i in range(1, N+1):
    log_v[i] = log_v[i-1] + k*(np.log(theta) - log_v[i-1])*dt + x*np.sqrt(dt) * np.random.normal()
volatility = np.sqrt(np.exp(log_v))  # μεταβληκοτητα
print(f"Volatility range: min = {np.min(volatility):.4f}, max = {np.max(volatility):.4f}, mean = {np.mean(volatility):.4f}")


# Building the binomial tree 
tree = np.zeros((N+1, N+1))
tree[0, 0] = S0
for i in range(N):
    sigma = volatility[i] # θετω απο το log-OU
    up = np.exp(sigma*np.sqrt(dt)) # πανω μερος του δεντρου
    down = 1 / up # κατω μερος του δεντρου
    prob = (np.exp(r*dt) - down) / (up - down) # Risk-neutral πιθανοτητα
    for j in range(i+1):
        if j == 0:
            tree[i+1, j] = tree[i, j] * down
        else:
            tree[i+1, j] = tree[i, j-1] * up # δημιουργεια κομβων στον δεντρο 


# Υπολογισμος της τιμης του option
option_price = np.zeros((N+1, N+1))
for j in range(N+1):
    option_price[N, j] = max(0, tree[N, j] - K)

for i in range(N - 1, -1, -1):
    sigma = volatility[i]
    up = np.exp(sigma*np.sqrt(dt))
    down = 1 / up
    prob = (np.exp(r*dt) - down) / (up - down)
    for j in range(i+1):
        option_price[i, j] = np.exp(-r * dt) * (prob * option_price[i + 1, j + 1] + (1 - prob) * option_price[i + 1, j])
    
# 6. Εκτύπωση αποτελέσματος
print(f"Call Option Price: {option_price[0, 0]:.2f}")

# 7. Οπτικοποίηση της προσομοιωμένης μεταβλητότητας
plt.plot(volatility)
plt.title('Log-Ornstein-Uhlenbeck Volatility')
plt.xlabel('Time Steps')
plt.ylabel('Volatility (σ)')
plt.show()
