import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="Analyse des Données Iris", layout="wide")

# Chargement des données
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Titre principal
st.title("🔍 Analyse des Données Iris")

# Aperçu des données
st.header("📋 Aperçu des données")
st.dataframe(df.head(10))  # Affichage limité aux 10 premières lignes

# Statistiques descriptives
st.header("📊 Statistiques descriptives")
st.write(df.describe())

# Visualisations interactives
st.header("🎨 Visualisations des données")

# Onglets pour les graphiques interactifs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Histogrammes", "Boxplots", "Scatter Plots", "Pairplots", "PCA"])

# Histogrammes
with tab1:
    st.subheader("Histogrammes")
    for col in df.columns[:-1]:
        fig, ax = plt.subplots()
        sns.histplot(df, x=col, hue='species', element="step", kde=True, ax=ax)
        st.pyplot(fig)

# Boxplots
with tab2:
    st.subheader("Boxplots")
    for col in df.columns[:-1]:
        fig, ax = plt.subplots()
        sns.boxplot(x='species', y=col, data=df, ax=ax)
        st.pyplot(fig)

# Scatter Plots interactifs
with tab3:
    st.subheader("Scatter Plots")
    col_x = st.selectbox("Axe X :", df.columns[:-1], index=0)
    col_y = st.selectbox("Axe Y :", df.columns[:-1], index=1)
    fig = px.scatter(df, x=col_x, y=col_y, color='species')
    st.plotly_chart(fig)

# Pairplots
with tab4:
    st.subheader("Pairplot")
    st.write("Les pairplots montrent les relations entre toutes les paires de variables.")
    fig = sns.pairplot(df, hue='species')
    st.pyplot(fig)

# PCA
with tab5:
    st.subheader("Analyse en Composantes Principales (PCA)")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.iloc[:, :-1])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    df_pca['species'] = df['species']
    fig = px.scatter(df_pca, x="PC1", y="PC2", color='species')
    st.plotly_chart(fig)

# Matrice de corrélation
st.header("🧬 Matrice de Corrélation")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.drop(columns=['species']).corr(), annot=True, cmap='coolwarm')
st.pyplot(fig)

# Footer
st.write("---")
st.write("© 2024 - Analyse des Données Iris • Par M. Merveille LIGAN")
