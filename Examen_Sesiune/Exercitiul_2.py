import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import geom

N = 10000
x = geom.rvs(0.3, size=N)
y = geom.rvs(0.5, size=N)
inside = x>y**2
pi = inside.sum()*4/N
outside = np.invert(inside)
plt.figure(figsize=(8, 8))
plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
plt.axis('square')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(x[inside], y[inside], 'b.')
plt.plot(x[outside], y[outside], 'r.')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Monte Carlo pentru P(X > Y^2)')
plt.show()