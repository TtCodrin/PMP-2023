import numpy as np
import arviz as az
from scipy import stats
import matplotlib.pyplot as plt
import pymc as pm

timpiAsteptare = stats.norm.rvs(4.0, 0.5, size=200) #generat 200 de timpi medii de asteptare
print(timpiAsteptare)

with pm.Model() as model:
    medieTimpiAsteptare = pm.Normal("medie", mu=4.0, sigma=0.5)
    trace = pm.sample(2000, tune = 1000)

pm.summary(trace)
az.plot_posterior(trace)
plt.show()