from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

model = BayesianNetwork([('Incendiu','Alarma'), ('Cutremur','Alarma'),('DeclansareAccidentalaAlarma','Alarma'),('Cutremur','Incendiu')])

cpd_cutremur = TabularCPD(variable="Cutremur", variable_card = 2, values=[[0.95],[0.05]])
cpd_incendiu = TabularCPD(variable="Incendiu", variable_card=2, values = [0.99],[0.01], evidence="Cutremur", evidence_card=[2])
cpd_declansare_accidentala_alarma = TabularCPD(variable="DeclansareAccidentalaAlarma", variable_card=2, values= [[0.99],[0.01]])
cpd_alarma=TabularCPD(variable="Alarma", variable_card=2, values=[0.99, 0.98, 0.05, 0.02],[0.01,0.02,0.95,0.98], evidence=["Cutremur","Incendiu","DeclansareAccidentalaAlarma", evidence_card=[2,2,2]])

model.add__cpds(cpd_cutremur, cpd_incendiu, cod_declansare_accidentala_alarma, cpd_alarma)