import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.feature_selection import f_regression, SelectKBest
# Ler os dados
df = pd.read_csv(r'C:\Users\Lfcgl\OneDrive\Desktop\BDS\diabetes_012_health_indicators_BRFSS2015.csv')

print(len(df))
print(len(df[df['Diabetes_012'] == 1]))
print(len(df[(df['BMI'] < 15) ]))
# df = df.drop(columns=['Sex', 'Fruits', 'Veggies', 'Education'])
df = df[(df['Diabetes_012'] != 1)]
df = df[(df['BMI'] < 80) & (df['BMI'] > 10)]
df['Diabetes_012'] = df['Diabetes_012'].replace(2, 1)
df = df.drop_duplicates(keep='first')



X = df.drop(['Diabetes_012'], axis=1)
y = df['Diabetes_012']
# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um modelo XGBoost
model = XGBClassifier()
model.fit(X_train, y_train)

# Usar SelectKBest com f_regression para avaliar a importância das features
selector = SelectKBest(f_regression, k='all')
selector.fit(X_train, y_train)

# Imprimir pontuações F e p-values
scores = pd.DataFrame({'Feature': X_train.columns, 'F-score': selector.scores_, 'p-value': selector.pvalues_})
print(scores.sort_values(by='F-score', ascending=False))



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# sm = SMOTE(random_state=42)
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# clf = XGBClassifier(gamma=0.1, colsample_bytree=0.75, subsample=0.7, max_depth=13, learning_rate=0.0001, n_estimators=250, random_state=42)
# clf.fit(X_train_res, y_train_res)


# y_pred = clf.predict(X_test)


# print(classification_report(y_test, y_pred))