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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve

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



models = [
    ('Decision Tree', dt_model),
    ('KNN', knn_model),
    ('Logistic Regression', logreg_model),
    ('Random Forest', rf_model),
    ('XGBoost', xgb_model),
    ('Naive Bayes', nb_model)
]

# Plotar curvas de aprendizado para cada modelo
for model_name, model in models:
    # Curva de aprendizado
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

    # Plotar a curva de aprendizado
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Treino')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Teste')
    plt.title(f'Curva de Aprendizado - {model_name}')
    plt.xlabel('Tamanho do Conjunto de Treino')
    plt.ylabel('Acurácia Média')
    plt.legend()
    plt.show()


for model_name, model in models:
    # Treinar o modelo
    model.fit(X_train, y_train)

    # Obter probabilidades previstas para a classe positiva
    y_score = model.predict_proba(X_test)[:, 1]

    # Calcular a curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plotar a curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.show()