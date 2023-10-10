import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

m1 = stats.expon(0,1/4).rvs(size=10000)
m2 = stats.expon(0,1/6).rvs(size=10000)

x=(m1+m2)/2

az.plot_posterior({'x':x})
plt.show() 