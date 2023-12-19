import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
import xgboost




df = pd.read_csv(r'C:\Users\Lfcgl\Desktop\BDS\diabetes_012_health_indicators_BRFSS2015.csv')
  
colunas = df.columns.tolist()


st.sidebar.title("Colunas em exibicao:")

colunas_exibidas = st.sidebar.multiselect("selecione as colunas:", colunas,default=colunas)

df_filtrado = df[colunas_exibidas]



conteiner = st.empty()

localnumero = st.empty()

valorslider = {}




with st.expander("Configurar Sliders"):
    for coluna in colunas:

        valores_inteiros = df[coluna].astype(int)
        min = valores_inteiros.min()
        max = valores_inteiros.max()

        valorslider[coluna] = st.slider(f"Selecione um valor para {coluna}", min, max, (min, max))


df_filtrado = df.copy()
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






with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.title("Análise de Diabetes")
st.write("Insira suas características para analisar sua chance de ter diabetes:")


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

    novo_individuo_scaled = scaler.transform(novo_individuo)

    with open('modelo_xgb.pkl', 'rb') as model_file:
        model_xgb = pickle.load(model_file)

    previsao_xgb = model_xgb.predict(novo_individuo_scaled)


    
    st.write("Resultado da previsão:")
    if previsao_xgb[0] != 0:
        st.write("Você tem chance de ter diabetes.")
    else:
        st.write("Você tem baixa chance de ter diabetes.")