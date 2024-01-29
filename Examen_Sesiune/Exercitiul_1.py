import pandas as pd
import numpy as np
import pymc as pm

#a)
data_frame = pd.read_csv('Titanic.csv');
data_frame['Age'].fillna(data_frame['Age'].median(), inplace=True)
embarked = data_frame['Embarked'].mode()[0]
data_frame['Embarked'].fillna(embarked, inplace=True)
data_frame['Fare'].fillna(data_frame['Fare'].mean(), inplace=True)
data_frame.drop(['Cabin'], axis=1, inplace=True)

print(data_frame.head())

#b)
p_class_values = data_frame['Pclass'].values
age_values = data_frame['Age'].values
survived_values = data_frame['Survived'].values

print(p_class_values)
print(age_values)

with pm.Model() as model:
    beta_p_class = pm.Normal('beta_p_class', mu=0)
    beta_age = pm.Normal('beta_age', mu=0)

    alpha = pm.Normal('alpha', mu=0)

    mu = alpha + beta_p_class * p_class_values + beta_age * age_values

    likelihood = pm.Bernoulli('likelihood', pm.math.sigmoid(mu), observed=survived_values)

    trace = pm.sample(2000, tune=1000)

pm.summary(trace)

predict_age = 30
predict_class = 2

data_to_predict = {'Age': predict_age, 'PClass': predict_class}

with model:
    predictions = pm.sample_posterior_predictive(trace, var_names=['Survived'], inputVals=[data_to_predict])

survival_probs = predictions['Survived']

hdi = pm.stats.hdi(survival_probs, hdi_prob=0.9)

print(hdi)
