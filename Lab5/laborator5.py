import pymc as pm
import pandas as pd

data_csv = pd.read_csv('trafic.csv')
data = data_csv.minut
increasing_hours = [7,16]
decreasing_hours = [8,19]

with pm.Model() as traffic_model:
    lambda_main = pm.Exponential("lambda_main", 1.0)
    lambda_arr=[]
    data_length=len(data_csv.minut)
    for i in data_csv.minut:
        hour = i // 60 + 4
        if hour in increasing_hours:
            lambda_arr.append(pm.Exponential("lambda"+str(i), 1.0))
        if hour in decreasing_hours:
            lambda_arr.append(pm.Exponential("lambda"+str(i), 1.0))

with traffic_model:
    total_trafic = pm.Poisson("traffic", lambda_arr, observed=data)
    trace = pm.sample(1000,tune=5000)

