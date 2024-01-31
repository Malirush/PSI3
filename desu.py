


train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

# Calcule as médias e desvios padrão dos escores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plote a curva de aprendizado
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="orange")
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Treino")
plt.plot(train_sizes, test_scores_mean, 'o-', color="orange", label="Validação")
plt.xlabel('Tamanho do Conjunto de Treinamento')
plt.ylabel('Pontuação de Precisão')
plt.title('Curva de Aprendizado do Modelo')
plt.legend(loc="best")
plt.show()





















sns.set_style("dark")
plt.style.use('dark_background')





plt.title(f'Distribuicao do IMC')
sns.boxplot(x=df['BMI'], color='purple')
plt.title(f'Boxplot IMC')
plt.show()







































# Se o número de clusters for 2 ou maior, você pode visualizar o Silhouette Plot
if num_clusters >= 2:

    # Calcular os valores de Silhouette para cada instância
    sample_silhouette_values = silhouette_samples(X_for_kmeans, X_test['Cluster'])

    # Criar um gráfico de Silhouette Plot
    plt.figure(figsize=(10, 8))

    y_lower = 10
    for i in range(num_clusters):
        # Agregar os valores de Silhouette para instâncias no cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[X_test['Cluster'] == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.viridis (float(i) / num_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Rótulo para o cluster no meio
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Calcule a próxima posição inferior
        y_lower = y_upper + 10  # 10 para os espaços em branco

    plt.title("Silhouette Plot para os Clusters - Silhouette Score: 0.52")
    plt.xlabel("Valores de Silhouette")
    plt.ylabel("Cluster")
    plt.show()

cluster_means = X_test.groupby('Cluster').mean()
print(cluster_means)
print(contingency_table)




plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
continuous_vars = ['BMI', 'PhysActivity', 'DiffWalk', 'Age', 'MentHlth', 'PhysHlth']
for i, var in enumerate(continuous_vars, 1):
    plt.subplot(3, 2, i) 
    sns.barplot(x=X_test['Cluster'], y=X_test[var], palette='viridis')
    plt.title(f'Média de {var} por Cluster')

plt.tight_layout()
plt.show()



































# analise graficos etc
cm = confusion_matrix(y_test, y_pred)
class_labels = ["Negativos", "Positivos"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='rocket', values_format='d')
plt.show()


plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()


plt.figure(figsize=(10, 8))
sns.histplot(df[df['Diabetes_012'] == 0]['Age'], color="purple", label="N/Diabetico", )
sns.histplot(df[df['Diabetes_012'] == 1]['Age'], color="pink", label="Diabetico", )
plt.title("Relação entre Idade e Diabetes")
plt.xlabel("Idade")
plt.ylabel("Contagem")
plt.legend()
plt.show()



plt.figure(figsize=(10, 8))
sns.histplot(df[df['Diabetes_012'] == 0]['BMI'], color="orange", label="N/Diabetico",kde=True )
sns.histplot(df[df['Diabetes_012'] == 1]['BMI'], color="red", label="Diabetico",kde=True )
plt.title("Relação entre IMC e Diabetes")
plt.xlabel("IMC")
plt.ylabel("Contagem")
plt.legend()
plt.show()



tbl_contingencia = pd.crosstab(df['Diabetes_012'], df['HighBP'], normalize='index')
plt.figure(figsize=(8, 6))
sns.heatmap(tbl_contingencia, annot=True, cmap='crest', fmt='0.2%', cbar=False)
plt.title('Diabetes vs Pressao Alta')
plt.show()



tbl_contingencia = pd.crosstab([df['HighChol'], df['HighBP']], df['Diabetes_012'], normalize='index')
plt.figure(figsize=(8, 6))
sns.heatmap(tbl_contingencia, annot=True, cmap='rocket', fmt=".2%", cbar=False)
plt.xlabel("Diabetes")
plt.ylabel("Colesterol - Pressao")
plt.title("Colesterol Alto & Pressao Alta vs Diabetes")
plt.show()


tbl_contingencia = pd.crosstab(df['Diabetes_012'], df['Stroke'], normalize='index')
plt.figure(figsize=(8, 6))
sns.heatmap(tbl_contingencia, annot=True, cmap='crest', fmt='0.2%', cbar=False)
plt.title('Diabetes vs Acidente Cerebral')
plt.show()



plt.figure(figsize=(8, 6))
sns.histplot(df[df['Diabetes_012'] == 0]['MentHlth'], color="red", label="Nao Diabetico" )
sns.histplot(df[df['Diabetes_012'] == 1]['MentHlth'], color="lightblue", label="Diabetico" )
plt.title("Diabetes vs Saude mental")
plt.show()