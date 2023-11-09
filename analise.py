import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



'''NOME DAS COLUNAS'''

diabetes = 'Diabetes_012'
saude_mental = 'MentHlth'
pressao_corpo = 'HighBP'
colesterol = 'HighChol'
exame_colesterol = 'CholCheck'
indice_massa_corporal = 'BMI'     #acima de 30 ja e perigoso
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
bd_diabetes = bd_diabetes[(bd_diabetes[indice_massa_corporal]>=15) & (bd_diabetes[indice_massa_corporal]<=35)]
bd_diabetes1total = bd_diabetes[(bd_diabetes[diabetes]==1)]
bd_diabetes2total = bd_diabetes[(bd_diabetes[diabetes]==2)]
bd_diabetes12 = bd_diabetes
quantidade = len(bd_diabetes12)
print("quatidade geral:",quantidade)



bd_diabetes12 = bd_diabetes12[(bd_diabetes[indice_massa_corporal]>=15) & (bd_diabetes[indice_massa_corporal]<=35)]
quantidade2 = len(bd_diabetes12)
print("quatidade apos primeiro filtro:",quantidade2)



'''
2- 24-30:  3- 30-35;  4-35-40 ; 5 - 40-45 ; 6- 45-50; 7- 50-55; 8- 55-60; 9- 60-65; 10- 65-70

55% fica entre os 50 e os 70 anos
'''
bd_diabetes2_30anosmais = bd_diabetes[(bd_diabetes[diabetes]>=1)&(bd_diabetes[idade]>=2)]

bd_diabetes2_entre50e70anos = bd_diabetes[(bd_diabetes[diabetes]>=1)&(bd_diabetes[idade]>=7)&(bd_diabetes[idade]<=14)]

db_naodiabetico = bd_diabetes[(bd_diabetes[diabetes]==0)]

db_naodiabeticopor2 = db_naodiabetico.iloc[:len(db_naodiabetico)//2]

db_diabetes12total = bd_diabetes[(bd_diabetes[diabetes]>=1)]

dados_completos = pd.concat([db_naodiabeticopor2, db_diabetes12total])
db_geralimc30 = bd_diabetes[(bd_diabetes[indice_massa_corporal]>=30)]
db_dbt12imc30 = bd_diabetes[(bd_diabetes[indice_massa_corporal]>=30)&(bd_diabetes[diabetes]>=1)]
#bd = bd_diabetes2_30anosmais[(bd_diabetes[dificuldade_andar]==0)]  #fumante com pre-diabetes



'''LENS/QUANTIDADES'''


dadosdiminuidos = len(dados_completos)

imc30maisgeral = len(db_geralimc30)
dbt12total = len(db_diabetes12total)
db0total = len(db_naodiabetico)
dbt1total = len(bd_diabetes1total)
dbt2total = len(bd_diabetes2total)
dbt12_30mais = len(bd_diabetes2_30anosmais)
dbt50_70anos = len(bd_diabetes2_entre50e70anos)
dbt12imc30mais = len(db_dbt12imc30)

total = len(bd_diabetes[diabetes])

filtrado = dbt12imc30mais
naofiltrado = dbt12total



X = bd_diabetes.drop('Diabetes_012', axis=1)
y = bd_diabetes['Diabetes_012']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Preprocessing: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an XGBoost model
model_xgb = xgb.XGBClassifier(n_estimators=100, random_state=42)
model_xgb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgb = model_xgb.predict(X_test)

# Evaluate the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Model Accuracy:", accuracy_xgb)




importancias_classes = model_xgb.feature_importances_

# Crie um DataFrame com as importâncias das características
importancias_df = pd.DataFrame({'Feature': X.columns, 'Importance': importancias_classes})

# Selecione as características mais importantes para cada classe (0, 1 e 2)
top_features_class_0 = importancias_df[importancias_df['Feature']].head(5)
top_features_class_1 = importancias_df[importancias_df['Feature']].head(5)
top_features_class_2 = importancias_df[importancias_df['Feature']].head(5)


# Imprima as características mais importantes para cada classe
print("Características mais importantes para a classe 0.0:")
print(top_features_class_0)

print("Características mais importantes para a classe 1.0:")
print(top_features_class_1)

print("Características mais importantes para a classe 2.0:")
print(top_features_class_2)

 

# Separar os dados em grupos com diabetes == 1 e diabetes == 2
dados_diabetes_0 = dados_completos[(dados_completos[diabetes] == 0)]
dados_diabetes_1 = dados_completos[(dados_completos[diabetes] == 1)]
dados_diabetes_2 = dados_completos[(dados_completos[diabetes] == 2)]
dados_diabetes_12 = dados_completos[(dados_completos[diabetes] >=1)]
dados_diabetes_12_imc28mais = dados_completos[(dados_completos[diabetes] >=1) & (dados_completos[indice_massa_corporal] >=28)]
dados_diabetes_12_idade40mais = dados_completos[(dados_completos[diabetes] >=1) & (dados_completos[idade] >=6)]

diabetes0_bmi30mais = dados_diabetes_1[indice_massa_corporal] >= 30
diabetes0_bmi30menos = dados_diabetes_1[indice_massa_corporal] <= 30




# Calcular a média para cada grupo em cada coluna
media_diabetes_1 = dados_diabetes_1.mean()
media_diabetes_2 = dados_diabetes_2.mean()

# Calcular a diferença entre as médias para cada coluna
diferenca_medias = media_diabetes_1 - media_diabetes_2

# Encontrar a coluna com a maior diferença
colunas_maior_diferenca = diferenca_medias.abs().nlargest(3).index

print("As 3 colunas com as maiores diferenças nas médias entre os grupos diabetes == 1 e diabetes == 2 são:")
print(colunas_maior_diferenca)


# Preparar os dados do IMC
imc_diabetes_0 = dados_diabetes_0[indice_massa_corporal]
imc_diabetes_1 = dados_diabetes_1[indice_massa_corporal]
imc_diabetes_2 = dados_diabetes_2[indice_massa_corporal]
imc_diabetes_12 = dados_diabetes_12[indice_massa_corporal]


# # Criar um boxplot
# plt.boxplot([imc_diabetes_0, imc_diabetes_2], labels=['Diabetes 1', 'Diabetes 2'])
# plt.title('Boxplot do IMC para sem Diabetes  e Diabetes 1/2')
# plt.xlabel('Grupo de Diabetes')
# plt.ylabel('IMC')
# plt.show()


'''a partir daqui e o peso das colunas'''


porcentagem_diabetes0_bmi30mais = (diabetes0_bmi30mais.sum() / len(dados_diabetes_1)) * 100
porcentagem_diabetes0_bmi30menos = (diabetes0_bmi30menos.sum() / len(dados_diabetes_1)) * 100

# Preparar os rótulos e porcentagens
categorias = [f"IMC > 30: {porcentagem_diabetes0_bmi30mais:.2f}%", f"IMC <= 30: {porcentagem_diabetes0_bmi30menos:.2f}%"]
porcentagens = [porcentagem_diabetes0_bmi30mais, porcentagem_diabetes0_bmi30menos]

# Criar o gráfico de barras
plt.bar(categorias, porcentagens)
plt.xlabel('Critérios de Filtro')
plt.ylabel('Porcentagem (%)')
plt.title('Porcentagem de Pessoas em Diferentes Grupos de Filtro no dados_diabetes_1')
plt.show()







diferencas_cv = []

# Itere sobre as colunas e calcule as diferenças nos valores do CV
for coluna in bd_diabetes.columns:
    media_diabetes = dados_diabetes_1[coluna].mean()
    media_sem_diabetes = dados_diabetes_0[coluna].mean()
    desvio_diabetes = dados_diabetes_1[coluna].std()
    desvio_sem_diabetes = dados_diabetes_0[coluna].std()
    
    cv_diabetes = desvio_diabetes / media_diabetes
    cv_sem_diabetes = desvio_sem_diabetes / media_sem_diabetes
    
    diferenca_cv = abs(cv_diabetes - cv_sem_diabetes)
    diferencas_cv.append((coluna, diferenca_cv))

# Ordene as colunas com base nas diferenças no CV
diferencas_cv = sorted(diferencas_cv, key=lambda x: x[1], reverse=True)

# Imprima as colunas mais distintas
for coluna, diferenca_cv in diferencas_cv:
    print(f'Coluna: {coluna}, Diferença no CV: {diferenca_cv}')















# porcentagem_diabetes_12_imc28mais = (len(dados_diabetes_12_imc28mais) / len(dados_diabetes_12)) * 100
# porcentagem_diabetes_12_idade40mais = (len(dados_diabetes_12_idade40mais) / len(dados_diabetes_12)) * 100

# # Preparar os rótulos e porcentagens
# categorias = [ f"IMC >= 28:  {porcentagem_diabetes_12_imc28mais:.2f}", f"Idade >= 40:  {porcentagem_diabetes_12_idade40mais:.2f}"]
# porcentagens = [porcentagem_diabetes_12_imc28mais, porcentagem_diabetes_12_idade40mais]

# # Criar o gráfico de barras
# plt.bar(categorias, porcentagens)
# plt.xlabel('Critérios de Filtro')
# plt.ylabel('Porcentagem (%)')
# plt.title('Porcentagem de Pessoas em Diferentes Grupos de Filtro')
# plt.show()







media_diabetes = dados_diabetes_0[indice_massa_corporal].mean()
max_diabetes = dados_diabetes_0[indice_massa_corporal].max()

print("MédiaBMI de diabetes0:", media_diabetes)
print("MáximoBMI de diabetes0:", max_diabetes)







