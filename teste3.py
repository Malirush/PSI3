import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Ler os dados
df = pd.read_csv(r'C:\Users\Lfcgl\OneDrive\Desktop\BDS\diabetes_012_health_indicators_BRFSS2015.csv')

# Pré-processamento
df = df.drop(columns=['Sex', 'Fruits', 'Veggies', 'Education'])
df = df[(df['Diabetes_012'] != 1)]
df = df[(df['BMI'] < 80) & (df['BMI'] > 10)]
df['Diabetes_012'] = df['Diabetes_012'].replace(2, 1)
df = df.drop_duplicates(keep='first')

X = df.drop(['Diabetes_012'], axis=1)
y = df['Diabetes_012']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Inicializar modelos
nb_model = GaussianNB()
dt_model = DecisionTreeClassifier(random_state=42) 
knn_model = KNeighborsClassifier()
logreg_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(gamma=0.1, colsample_bytree=0.75, subsample=0.7, max_depth=13, learning_rate=0.0001, n_estimators=250, random_state=42)

# Treinar modelos
dt_model.fit(X_train_res, y_train_res) 
nb_model.fit(X_train_res, y_train_res)
knn_model.fit(X_train_res, y_train_res)
logreg_model.fit(X_train_res, y_train_res)
rf_model.fit(X_train_res, y_train_res)
xgb_model.fit(X_train_res, y_train_res)

# Fazer previsões
dt_pred = dt_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
logreg_pred = logreg_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
nb_pred = nb_model.predict(X_test)

# # Avaliar modelos
# print("\nNaive Bayes Classification Report:")
# print(classification_report(y_test, nb_pred)) 

# print("\nKNN Classification Report:")
# print(classification_report(y_test, knn_pred))

# print("\nLogistic Regression Classification Report:")
# print(classification_report(y_test, logreg_pred))

# print("\nRandom Forest Classification Report:")
# print(classification_report(y_test, rf_pred))

# print("\nXGBoost Classification Report:")
# print(classification_report(y_test, xgb_pred))

# print("\nDecision Tree Classification Report:")
# print(classification_report(y_test, dt_pred))

