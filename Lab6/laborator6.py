import numpy as np
import pymc as pm
import arviz as az

yValues = np.array([0,5,10])
thetaValues = np.array([0.2, 0.5])

for y in yValues:
    for t in thetaValues:
        with pm.Model() as model:
            n = pm.Poisson("n", 10)
            b = pm.Binomial("y", n = n, p = t, observed = y)
            data = pm.sample(1000)

        az.plot_posterior(data)
