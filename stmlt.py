import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
from PIL import Image



df = pd.read_csv(r'C:\Users\Lfcgl\OneDrive\Desktop\BDS\diabetes_012_health_indicators_BRFSS2015.csv')
  
colunas = df.columns.tolist()



# Adicione suas abas
tab_options = ["Analise! - ", "Analises Realizadas - ", "Modelo ML - "]
selected_tab = st.sidebar.radio("Diabeat:", tab_options)



# Conteúdo para cada aba
if selected_tab == "Analise! - ":
    
    st.title("Analise você mesmo:")

    with st.expander("Opções de Coluna"):
        colunas_exibidas = st.multiselect("Selecione as colunas:", colunas, default=colunas)
    df_filtrado = df[colunas_exibidas]
    colunas = df_filtrado.columns.to_list()
    conteiner = st.empty()
    localnumero = st.empty()
    valorslider = {}
    with st.expander("Configurar Sliders"):
        for coluna in colunas:
            valores_inteiros = df_filtrado[coluna].astype(int)
            min = valores_inteiros.min()
            max = valores_inteiros.max()
            valorslider[coluna] = st.slider(f"Selecione um valor para {coluna}", min, max, (min, max))
    for coluna, valores_slider in valorslider.items():
        df_filtrado = df_filtrado[(df_filtrado[coluna] >= valores_slider[0]) & (df_filtrado[coluna] <= valores_slider[1])]

    numerototal=len(df_filtrado)
    localnumero.write(f"Numero de individuos : {numerototal}")
    conteiner.write(df_filtrado)

    fig, ax = plt.subplots()
    contagem_diabetes = df_filtrado["Diabetes_012"].value_counts().sort_index()
    contagem_diabetes.plot(kind="bar", ax=ax)
    total = contagem_diabetes.sum()
    porcentagens = [f"({count / total * 100:.2f}%)" for count in contagem_diabetes]
    for i, (v, porcentagem) in enumerate(zip(contagem_diabetes, porcentagens)):
        ax.text(i, v + 0.1, f"{v} {porcentagem}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Contagem de Diabetes")
    ax.set_xlabel("Valores de Diabetes")
    ax.set_ylabel("Contagem")
    st.pyplot(fig)

    colunas = df_filtrado.columns
    coluna_selecionada = st.expander("Configurar Box Plot").selectbox("Selecione uma coluna", colunas)
    fig, ax = plt.subplots()
    df_filtrado.boxplot(column=coluna_selecionada, ax=ax)
    ax.set_title(f"Box Plot - {coluna_selecionada}")
    st.pyplot(fig)





















   
elif selected_tab == "Analises Realizadas - ":
    st.title('Avaliaçoes:')
    
    
    fscore_data = {
        'Colunas': ['GenHlth', 'HighBP', 'BMI', 'HighChol', 'DiffWalk', 'Age', 'HeartDiseaseorAttack', 'PhysHlth', 'Income', 'Stroke', 'CholCheck', 'PhysActivity', 'HvyAlcoholConsump', 'AnyHealthcare', 'Smoker', 'MentHlth', 'NoDocbcCost'],
        'F-Score': [13757.49, 13076.88, 8731.97, 7619.25, 7115.08, 6898.82, 4967.35, 3525.71, 2742.78, 1576.97, 1431.51, 1354.76, 1228.39, 312.27, 235.54, 178.08, 1.07],
        'Valor de probabilidade': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.59e-312, 1.64e-295, 3.09e-268, 7.96e-70, 3.96e-53, 1.33e-40, 0.301]
    }
    data_means = {
        'HighBP': [0.447542, 0.531849, 0.612135],
        'HighChol': [0.433315, 0.524548, 0.544120],
        'CholCheck': [0.950055, 0.959664, 0.976174],
        'BMI': [28.644923, 30.067896, 29.942236],
        'Smoker': [0.456350, 0.589159, 0.564344],
        'Stroke': [0.040375, 0.083409, 0.106663],
        'HeartDiseaseorAttack': [0.099428, 0.165359, 0.213603],
        'PhysActivity': [0.738909, 0.581675, 0.550769],
        'HvyAlcoholConsump': [0.074110, 0.067348, 0.038925],
        'AnyHealthcare': [0.936151, 0.918233, 0.951517],
        'NoDocbcCost': [0.086677, 0.241650, 0.144203],
        'GenHlth': [2.464352, 3.491513, 3.682782],
        'MentHlth': [1.538992, 26.616901, 2.658678],
        'PhysHlth': [1.358536, 13.815112, 24.697326],
        'DiffWalk': [0.135469, 0.450082, 0.560188],
        'Age': [8.017386, 7.475269, 9.058180],
        'Income': [5.926005, 4.711261, 4.948746]
    }
    data_contingency = {'Diabetes_012': [0, 1, 0, 1, 0, 1],
        'Cluster': [0, 0, 1, 1, 2, 2],
        'Count': [36381, 6987, 4222, 1257, 5199, 2020]}
    data_classification_report = {
        'precision': [0.88, 0.47],
        'recall': [0.89, 0.43],
        'f1-score': [0.88, 0.45],
        'support': [45802, 10264]
    }


    df_fscore = pd.DataFrame(fscore_data)
    def formatar_valor(valor):
        if isinstance(valor, str):
            return valor
        if valor>1:
            return f"{valor:.2f}"
        return f"{valor:.2e}"
    df_fscore= df_fscore.applymap(formatar_valor)

    df_contingency = pd.DataFrame(data_contingency)
    contingency_table = df_contingency.pivot_table(values='Count', index='Cluster', columns='Diabetes_012', aggfunc='sum', fill_value=0, )

    df_classification_report = pd.DataFrame(data_classification_report, index=['0.0', '1.0'])
    df_classification_report.index.name = 'Diabetes_012'

    df_cluster_means = pd.DataFrame(data_means)
    df_cluster_means.index.name = 'Cluster'

 
    st.write(' - F-score das Features:')
    st.table(df_fscore)

    st.write(' - Matriz de correlação:')
    st.write(' "Relacao alta entre colunas sobre saude em geral"   ')
    st.image("./analises-final/matrizcorrela.png")

    st.write(" - Tabela de Métricas de Classificação: Acertividade geral - 0.81")
    st.table(df_classification_report)

    st.write(' - Matriz de confusão:')
    st.image("./analises-final/marizconfusao.png")

    st.write(" - Curva de aprendizado do modelo:")
    st.write(' "Pontuaçã boa com validação estável (baixo ou nulo overfitting)"   ')
    st.image("./analises-final/curva-aprendizado.png")


    st.write(' - Silhouette plot dos Clusters:')
    st.write(' "Pequena perca nos cluster 1 e 2 porem com estabilidade boa"   ')
    st.image("./analises-final/silhouetteplot.png")

    st.write(' - Medias nos Clusters:')
    st.write(' "Evidenciando a clara diferenca em saude no geral entre os individuos dos clusters, sendo 0 mais saudáveis, 1 médios e 2 os mais debilidados"   ')
    st.table(df_cluster_means)
    st.image("./analises-final/medias-clusters.png")
    
    st.write(" - Tabela de Contingência dos clusters:")
    st.write(' "Num. individuos vs Diabetes"   ')
    st.table(contingency_table)
    st.image("./analises-final/diabetes-cluster.png")


    st.title('Analises Gerais:')

    st.write(' - Boxplot - IMC :')
    st.write(' "Percepção seguida da retirada de outliers com IMC acima de 80"   ')
    st.image("./analises-final/boxplot-imc.png")

    st.write(' - Diabetes vs IMC :')
    st.write(' "Aumento significativo e estabilidade em IMC acima de 27 nos diabeticos"   ')
    st.image("./analises-final/diabetes-imc.png")

    st.write(' - Diabetes vs Idade :')
    st.write(' "Devido ao envelhecimento o corpo possui funçoes afetadas, uma delas a regulação da glicose "   ')
    st.image("./analises-final/diabetes-idade.png")

    st.write(' - Diabetes vs Pressão alta:')
    st.write(' "Devido ao alto nivel glicemico no sangue, a pressão sanguinea fica debilitada"   ')
    st.image("./analises-final/diabetes-pressaoalta.png")

    st.write(' - Diabetes vs Pressão alta & Colesterol alto:')
    st.image("./analises-final/colesterolpressao-diabetes.png")

    st.write(' - Diabetes vs Acidente cerebral:')
    st.write(' "Aumento significativo no numero de diabeticos, considerando tambem a diferença no numero de individuos em cada ocasião"   ')
    st.image("./analises-final/acidcerebral-diabetes.png")

    st.write(' - Diabetes vs Saude mental:')
    st.write(' "Crescimento relacionado, devido a depressão o individuo acaba se alimentando de forma erronia e não busca a realização de atividades fisicas por exemplo, causando um aumento nos casos."   ')
    st.image("./analises-final/saudemental-diabetes.png")















elif selected_tab == "Modelo ML - ":
     

    st.title("Análise de Diabetes:")
    st.write("Insira suas características para analisar sua chance de ter diabetes:")

    df = df.drop(columns=['Sex', 'Fruits', 'Veggies', 'Education'])
    df = df[(df['Diabetes_012'] != 1)]
    df = df[(df['BMI'] < 80) & (df['BMI'] > 10)]
    df['Diabetes_012'] = df['Diabetes_012'].replace(2, 1)
    df = df.drop_duplicates(keep='first')   


    with st.expander("Configurar Sliders"):
        valorslider = {}  
        for coluna in df.columns:
            if coluna != 'Diabetes_012': 
                valores_inteiros = df[coluna].astype(int)
                min_valor = valores_inteiros.min()
                max_valor = valores_inteiros.max()
                key = f"slider_{coluna}"
                valorslider[coluna] = st.slider(f"Selecione um valor para {coluna}", min_valor, max_valor, min_valor, key=key)

    if st.button("Analisar Chance de Diabetes"):
        novo_individuo = pd.DataFrame({coluna: [float(valor_slider)] for coluna, valor_slider in valorslider.items()})
        with open('modelo_xgboost.pkl', 'rb') as model_file:
            model_xgb = pickle.load(model_file)
        previsao_xgb = model_xgb.predict(novo_individuo)
        st.write("Resultado da previsão:")
        if previsao_xgb[0] != 0:
            st.write("Você tem chance de ter diabetes.")
        else:
            st.write("Você tem baixa chance de ter diabetes.")













