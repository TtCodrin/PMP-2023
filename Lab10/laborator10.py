import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')
dummy_data = np.loadtxt('./data/dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/
x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')

with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
    e = pm.HalfNormal('e', 5)
    u = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=u, sigma=e, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

#1a)
with model_p:
    pm.set_data({'x_1s': x_1s})
    y_pred = pm.sample_posterior_predictive(idata_p, 2000)['y_pred']
    pm.plot_posterior_predictive(idata_p, var_names=["y_pred"], mean=True)
    plt.show()

#1b)
with pm.Model() as model_p_sd_100:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
    e = pm.HalfNormal('e', 5)
    u = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=u, sigma=e, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

with model_p_sd_100:
    pm.set_data({'x_1s': x_1s})
    y_pred = pm.sample_posterior_predictive(idata_p, 2000)['y_pred']
    pm.plot_posterior_predictive(idata_p, var_names=["y_pred"], mean=True)
    plt.show()

sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
with pm.Model() as model_p_array:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=sd, shape=order)
    e = pm.HalfNormal('e', 5)
    u = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=u, sigma=e, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

with model_p_array:
    pm.set_data({'x_1s': x_1s})
    y_pred = pm.sample_posterior_predictive(idata_p, 2000)['y_pred']
    pm.plot_posterior_predictive(idata_p, var_names=["y_pred"], mean=True)
    plt.show()

#2)
data_x = np.random.rand(500,2)
data_y = np.random.rand(500,2)

x_1p = np.vstack([data_x**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/
x_1p.std(axis=1, keepdims=True)
y_1s = (data_y - data_y.mean()) / data_y.std()

with pm.Model() as model_p_sd_100_500_values:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
    e = pm.HalfNormal('e', 5)
    u = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=u, sigma=e, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

with model_p_sd_100_500_values:
    pm.set_data({'x_1s': x_1s})
    y_pred = pm.sample_posterior_predictive(idata_p, 2000)['y_pred']
    pm.plot_posterior_predictive(idata_p, var_names=["y_pred"], mean=True)
    plt.show()

sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
with pm.Model() as model_p_array_500_values:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=sd, shape=order)
    e = pm.HalfNormal('e', 5)
    u = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=u, sigma=e, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

with model_p_array_500_values:
    pm.set_data({'x_1s': x_1s})
    y_pred = pm.sample_posterior_predictive(idata_p, 2000)['y_pred']
    pm.plot_posterior_predictive(idata_p, var_names=["y_pred"], mean=True)
    plt.show()

#3)
order = 3 
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/
x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_cubic:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
    e = pm.HalfNormal('e', 5)
    u = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=u, sigma=e, observed=y_1s)
    idata_c = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_patratic:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=3)
    e = pm.HalfNormal('e', 5)
    u = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=u, sigma=e, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_liniar:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    e = pm.HalfNormal('e', 5)
    u = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=u, sigma=e, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True)

waic_value_c = az.waic(idata_c)
loo_value_c = az,loo(idata_c)
print("waic cubic: ", waic_value_c)
print("loo cubic: ", loo_value_c)

waic_value_p = az.waic(idata_p)
loo_value_p = az,loo(idata_p)
print("waic patratic: ", waic_value_p)
print("loo patratic: ", loo_value_p)

waic_value_l = az.waic(idata_l)
loo_value_l = az,loo(idata_l)
print("waic liniar: ", waic_value_l)
print("loo liniar: ", loo_value_l)

with model_cubic:
    pm.set_data({'x_1s': x_1s})
    y_pred = pm.sample_posterior_predictive(idata_c, 2000)['y_pred']
    pm.plot_posterior_predictive(idata_c, var_names=["y_pred"], mean=True)
    plt.show()

with model_patratic:
    pm.set_data({'x_1s': x_1s})
    y_pred = pm.sample_posterior_predictive(idata_p, 2000)['y_pred']
    pm.plot_posterior_predictive(idata_p, var_names=["y_pred"], mean=True)
    plt.show()

with model_liniar:
    pm.set_data({'x_1s': x_1s})
    y_pred = pm.sample_posterior_predictive(idata_l, 2000)['y_pred']
    pm.plot_posterior_predictive(idata_l, var_names=["y_pred"], mean=True)
    plt.show()