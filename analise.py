import pandas as pd
import matplotlib.pyplot as plt




'''NOME DAS COLUNAS'''

diabetes = 'Diabetes_012'
saude_mental = 'MentHlth'
pressao_corpo = 'HighBP'
colesterol = 'HighChol'
exame_colesterol = 'CholCheck'
indice_massa_corporal = 'BMI'
fumante = 'Smoker'
acidente_vascular_cerebral = 'Stroke'
heart_attack = 'HeartDiseaseorAttack'
atv_fisica = 'PhysActivity'
frutas = 'Fruits'
vegetais = 'Veggies'
muito_alcool = 'HvyAlcoholConsump'
cubertura_saude = 'AnyHealthcare'
dificul_pagament_medico = 'NoDocbcCost'
saude_geral = 'GenHlth'
saude_mental_ruim_30dias = 'MentHlth'
saude_corpo_ruim_30dias = 'PhysHlth'
dificuldade_andar = 'DiffWalk'
sexo = 'Sex'
idade= 'Age'
educacao = 'Education'
ganho = 'Income'



'''CLASSIFICACAO POR ESTAGIO DA DIABETES'''
bd_diabetes= pd.read_csv(r'C:\Users\Lfcgl\Desktop\BDS\diabetes_012_health_indicators_BRFSS2015.csv')
bd_diabetes= bd_diabetes.sort_values(by=diabetes)



'''FILTROS'''
bd_diabetes1total = bd_diabetes[(bd_diabetes[diabetes]==1)]
bd_diabetes2total = bd_diabetes[(bd_diabetes[diabetes]==2)]

'''
2- 24-30:  3- 30-35;  4-35-40 ; 5 - 40-45 ; 6- 45-50; 7- 50-55; 8- 55-60; 9- 60-65; 10- 65-70

55% fica entre os 50 e os 70 anos
'''
bd_diabetes2_30anosmais = bd_diabetes[(bd_diabetes[diabetes]==2)&(bd_diabetes[idade]>=7)&(bd_diabetes[idade]<=10)]




bd = bd_diabetes2_30anosmais[(bd_diabetes[dificuldade_andar]==0)]  #fumante com pre-diabetes



'''LENS/QUANTIDADES'''

dbt1total = len(bd_diabetes1total)
dbt2total = len(bd_diabetes2total)
dbt230mais = len(bd_diabetes2_30anosmais)
total = len(bd_diabetes[diabetes])

filtrado = dbt230mais





'''CONFIG DO GRAFICO'''
categorias = [f'total:{total}',f'filtrado:{filtrado}',f'NaoFiltrado:{dbt2total}']
qntd  = [total, filtrado,dbt2total]
diferenca = (filtrado / dbt2total) * 100
plt.bar(categorias, qntd)
plt.bar(categorias, qntd)
plt.xlabel('Categorias')
plt.ylabel('Quantidade')
plt.title(f'Comparação de Quantidades : {diferenca:.2f}%')
plt.show()











