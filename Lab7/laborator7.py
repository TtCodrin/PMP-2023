import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('auto-mpg.csv')

data = data.dropna(subset=['horsepower','mpg'])

plt.figure(figsize=(15, 15))
plt.scatter(data['horsepower'], data['mpg'], alpha=0.7)
plt.title('horsepower și mpg')
plt.xlabel('horsepower')
plt.xticks(rotation=45, fontsize = 6)
plt.ylabel('mpg')
plt.grid(True)
plt.show()
