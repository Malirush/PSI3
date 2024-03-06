import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

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
models = {
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(gamma=0.1, colsample_bytree=0.75, subsample=0.7, max_depth=13, learning_rate=0.0001, n_estimators=250, random_state=42)
}

# Criar um DataFrame para armazenar os resultados
results_df = pd.DataFrame(columns=['Modelo', 'Acurácia', 'F1 Score'])

# Treinar e avaliar modelos
for model_name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results_df = results_df.append({'Modelo': model_name, 'Acurácia': accuracy, 'F1 Score': f1}, ignore_index=True)

# Imprimir os resultados
print(results_df)

# Exportar para um arquivo CSV (opcional)
results_df.to_csv('resultados_modelos.csv', index=False)