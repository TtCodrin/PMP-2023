import scipy.stats

l=20
standard_deviation = 0.5
order_time_mean = 2
alpha=10

poisson_d = scipy.stats.poisson(l)
normal_d = scipy.stats.norm(loc= order_time_mean,scale= standard_deviation)
exponential_d = scipy.stats.expon(scale=1/alpha)

numar_clienti = poisson_d.rvs()
timp_comanda = normal_d.rvs()
timp_gatire = exponential_d.rvs()


print(f"Alpha ={alpha}")
print(f"Numar clienti: {numar_clienti}")
print(f"Timp plasare comanda: {timp_comanda}")
print(f"Timp gatire: {timp_gatire}")




