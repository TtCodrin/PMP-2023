import numpy as np
import arviz as az

clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster),
np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix))

array_clusters=[2, 3, 4]
models = []
for cluster in array_clusters:
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        z = pm.Categorical('z', p=p, shape=len(mix))
        means = pm.Normal('means', mu=mix.mean(), sigma=10, shape=cluster)
        sd=pm.HalfNormal('sd', sigma=10)
        y=pm.Normal('y', mu=means[z], sigma=sd, observed = mix)
        idata_kg=pm.sample(return_inferencedata=True)
        
        waic = az.waic(idata_kg)
        loo = az.loo(idata_kg)
        print(f"WAIC: {waic} | LOO: {loo}")