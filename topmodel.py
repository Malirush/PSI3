import pandas as pd


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import f_regression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

import pickle
import numpy as np


#ocorrencias diabete0-213703  - apos retirar duplicadas 153363
#ocorrencias diabete1-4631    - retirado do BD
#ocorrencias diabete2-35346   - apos retirar duplicadas 33521

#DF total  - 186884






# csv
df= pd.read_csv(r'C:\Users\Lfcgl\OneDrive\Desktop\BDS\diabetes_012_health_indicators_BRFSS2015.csv')

df = df.drop(columns=['Sex', 'Fruits', 'Veggies', 'Education'])
df = df[(df['Diabetes_012'] != 1)]
df = df[(df['BMI'] < 80) & (df['BMI'] > 10)]
df['Diabetes_012'] = df['Diabetes_012'].replace(2, 1)
df = df.drop_duplicates(keep='first')



# modelo
X = df.drop('Diabetes_012', axis=1)
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


y_pred = clf.predict(X_test)











X_test['Cluster'] = y_pred
X_for_kmeans = X_test.drop('Cluster', axis=1)
num_clusters = 3  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
X_test['Cluster'] = kmeans.fit_predict(X_for_kmeans)
cluster_counts = X_test['Cluster'].value_counts()
contingency_table = pd.crosstab(X_test['Cluster'], y_test)

silhouette_avg = silhouette_score(X_for_kmeans, X_test['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")



print(classification_report(y_test, y_pred))















