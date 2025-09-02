import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"Δεν βρέθηκαν δεδομένα για το ticker {ticker}. Ελέγξτε το όνομα.")
    
    S0 = df['Close'].iloc[-1].item()
    prices = df['Close']
    print(f"\nInitial stock price (S0) για {ticker}: {S0:.2f}")
except Exception as e:
    print(f"Error fetching data: {e}")
    S0 = 150.0  # Αν είναι κενό βάζουμε μια συγκεκριμένη τιμή
    print(f"\nUsing fallback stock price: {S0}")

# Υπολογισμός κ, μ, θ 


# Παραμετροι
k = 2.0  # Ποσό γρήγορα επιστρέφει στο μέσο όρο μετά την απόκλιση
theta = 0.04  # Μέση τιμή της διακύμανσης
x = 0.1  # Volatility of volatility
v0 = 0.04  # Αρχική τιμή της διακύμανσης
print(f"\nParameters used (manually set): k = {k:.4f}, theta = {theta:.4f}, x = {x:.4f}\n")

# Παράμετροι για το δέντρο
T = 1.0  # Χρόνος λήξης
dt = T / N  # Χρονικά διαστήματα βασισμένα στο N
r = 0.05  # Ρίσκο ελεύθερο επιτόκιο
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