import arviz as az
from matplotlib import pyplot as plt

centered_eight_data = az.load_arviz_data("centered_eight")

non_centered_eight_data = az.load_arviz_data("non_centered_eight")

#ex1
print("")
print("Modelul centrat:")
print("Numar lanturi: ",centered_eight_data.posterior.chain.size)
print("Marimea totala a esantionului: ", centered_eight_data.posterior.draw.size)
print("")

az.plot_posterior(centered_eight_data)

print("Modelul necentrat: ")
print("Numar lanturi: ", non_centered_eight_data.posterior.chain.size)
print("Marimea totala a esantionului: ", non_centered_eight_data.posterior.draw.size)
print("")
az.plot_posterior(non_centered_eight_data)

plt.show()

#ex2
centered_rhat = az.rhat(centered_eight_data, var_names=["mu", "tau"])
non_centered_rhat = az.rhat(non_centered_eight_data, var_names=["mu", "tau"])

print("Rhat model centrat:")
print(centered_rhat)

print("Rhat model necentrat:")
print(non_centered_rhat)

az.plot_autocorr(centered_eight_data, var_names=["mu","tau"], combined=True)
az.plot_autocorr(non_centered_eight_data, var_names=["mu","tau"], combined=True)

plt.show()

#ex3
centered_divergences = centered_eight_data.sample_stats.diverging.sum()

non_centered_divergences = non_centered_eight_data.sample_stats.diverging.sum()

print("Numar divergente pentru modelul centrat: ", centered_divergences)
print("Numar divergente pentru modelul necentrat: ", non_centered_divergences)

az.plot_pair(centered_eight_data, var_names=["mu","tau"], divergences=True)
az.plot_pair(non_centered_eight_data, var_names=["mu","tau"], divergences=True)

plt.show()
