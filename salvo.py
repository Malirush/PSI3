#Separe as características (X) e o alvo (y)
X = dados_completos.drop('Diabetes_012', axis=1)
y = dados_completos['Diabetes_012']

 # Treinar um modelo Random Forest para obter importância das características
modelo_rf = RandomForestClassifier(n_estimators=100)  # Pode ajustar o número de árvores
modelo_rf.fit(X, y)

# Obter a importância das características
importancias = modelo_rf.feature_importances_

# Crie um DataFrame com a importância das características e seus nomes
importancias_df = pd.DataFrame({'Feature': X.columns, 'Importance': importancias})

# Classifique as características por importância em ordem decrescente
importancias_df = importancias_df.sort_values(by='Importance', ascending=False)

# Selecione as 8 características mais importantes
caracteristicas_selecionadas = importancias_df.head(11)
print(caracteristicas_selecionadas)


# Use as características selecionadas para treinar um novo modelo de regressão logística
X_selecionado = X[caracteristicas_selecionadas['Feature']]
X_train, X_test, y_train, y_test = train_test_split(X_selecionado, y, test_size=0.4, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo com características selecionadas:", accuracy)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))