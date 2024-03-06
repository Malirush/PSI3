import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import f_regression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle

# Ler os dados
df = pd.read_csv(r'C:\Users\Lfcgl\OneDrive\Desktop\BDS\diabetes_012_health_indicators_BRFSS2015.csv')

# Pré-processamento
df = df.drop(columns=['Sex', 'Fruits', 'Veggies', 'Education'])
df = df[(df['Diabetes_012'] != 1)]
df = df[(df['BMI'] < 80) & (df['BMI'] > 10)]
df['Diabetes_012'] = df['Diabetes_012'].replace(2, 1)
df = df.drop_duplicates(keep='first')


num_clusters = 3  
X_for_kmeans = df.drop('Diabetes_012', axis=1)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_for_kmeans)

model_data = {
    'kmeans_model': kmeans,
    'clusters': df['Cluster']
 }




with open('modelo_kmeans.pkl', 'wb') as file:
    pickle.dump(model_data, file)


silhouette_avg = silhouette_score(X_for_kmeans, df['Cluster'])
print(f"\nNúmero de Clusters: {num_clusters}")
print(f"Silhouette Score: {silhouette_avg}")


X = df.drop(['Diabetes_012','Cluster'], axis=1)
y = df['Diabetes_012']

fscore, pontos = f_regression(X, y)
f_reg = pd.DataFrame({'Colunas': X.columns, 'F-Score': fscore, 'Valor de probabilidade': pontos})
f_reg = f_reg.sort_values(by='F-Score', ascending=False)
print(f_reg)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
clf = XGBClassifier(gamma=0.1, colsample_bytree=0.75, subsample=0.7, max_depth=13, learning_rate=0.0001, n_estimators=250, random_state=42)
clf.fit(X_train_res, y_train_res)

with open('modelo_xgboost.pkl', 'wb') as file:
     pickle.dump(clf, file)

y_pred = clf.predict(X_test)

# # Previsão do novo indivíduo
# novo_individuo = pd.DataFrame(...)  # Substitua ... pelos dados do novo indivíduo
# novo_individuo = novo_individuo[X.columns]  # Certifique-se de que as colunas são as mesmas usadas no treinamento
# novo_cluster = kmeans.predict(novo_individuo)

# print(f"O novo indivíduo se encaixa no Cluster: {novo_cluster}")

# Restante do seu código
print(classification_report(y_test, y_pred))