import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("data/features/vn30/VCB.csv")
print(df.head())
print(df.info())
print(df.describe())


plt.figure(figsize=(12,5))
plt.plot(df['date'], df['close'], label='Close')
plt.plot(df['date'], df['ma_5'], label='MA 5')
plt.legend()
plt.title("Price & Trend")
plt.show()



plt.figure(figsize=(12,5))
plt.plot(df['date'], df['return_1d'], label='Return 1D')
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.title("Daily Return")
plt.show()

plt.figure(figsize=(12,5))
plt.plot(df['date'], df['volatility_10'], label='Volatility 10D')
plt.legend()
plt.title("Rolling Volatility")
plt.show()


plt.figure(figsize=(8,4))
plt.hist(df['return_1d'], bins=50)
plt.title("Return Distribution")
plt.show()

plt.figure(figsize=(4,6))
plt.boxplot(df['return_1d'], vert=True)
plt.title("Return Boxplot")
plt.show()
