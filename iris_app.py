import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="Analyse complète des Iris", layout="wide")

# Chargement des données
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Titre principal
st.title("🔍 Analyse complète des Données Iris")

# ---- Section 1 : Aperçu des données ---- #
st.header("📋 Aperçu des données")
st.dataframe(df)

# ---- Section 2 : Statistiques descriptives ---- #
st.header("📊 Statistiques descriptives")
st.write(df.describe())

# ---- Section 3 : Visualisations interactives ---- #
st.header("🎨 Visualisations des données")

# ---- Graphiques interactifs avec onglets ---- #
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Histogrammes", "Boxplots", "Scatter Plots", "Pairplots", "PCA"])

# ---- Histogrammes ---- #
with tab1:
    st.subheader("Histogrammes")
    for col in df.columns[:-1]:
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(df, x=col, hue='species', element="step", palette='husl', kde=True, ax=ax_hist)
        st.pyplot(fig_hist)

# ---- Boxplots ---- #
with tab2:
    st.subheader("Boxplots")
    for col in df.columns[:-1]:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x='species', y=col, data=df, palette='Set2', ax=ax_box)
        st.pyplot(fig_box)

# ---- Scatter Plots ---- #
with tab3:
    st.subheader("Scatter Plots interactifs")
    col_x = st.selectbox("Choisissez l'axe X :", df.columns[:-1], index=0)
    col_y = st.selectbox("Choisissez l'axe Y :", df.columns[:-1], index=1)
    fig_scatter = px.scatter(df, x=col_x, y=col_y, color='species', title=f"{col_x} vs {col_y}")
    st.plotly_chart(fig_scatter)

# ---- Pairplots ---- #
with tab4:
    st.subheader("Pairplot")
    fig_pairplot = sns.pairplot(df, hue='species', palette='husl')
    st.pyplot(fig_pairplot)

# ---- PCA ---- #
with tab5:
    st.subheader("Analyse en Composantes Principales (PCA)")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.iloc[:, :-1])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    df_pca['species'] = df['species']
    
    fig_pca = px.scatter(df_pca, x="PC1", y="PC2", color='species', title="Projection PCA des données Iris")
    st.plotly_chart(fig_pca)

# ---- Matrice de corrélation ---- #
st.header("🧬 Matrice de Corrélation")
# Exclusion de la colonne catégorielle
corr_matrix = df.drop(columns=['species']).corr()
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
st.pyplot(fig_corr)

# ---- Heatmap des variables ---- #
st.subheader("Heatmap des caractéristiques par espèce")
pivot_data = df.melt(id_vars='species', var_name='Caractéristique', value_name='Valeur')
fig_heat = px.density_heatmap(pivot_data, x='species', y='Caractéristique', z='Valeur', color_continuous_scale='viridis')
st.plotly_chart(fig_heat)

# ---- Graphique en violon ---- #
st.header("🎻 Violin Plots")
selected_violin_feature = st.selectbox("Choisissez une caractéristique pour le Violin Plot :", df.columns[:-1])
fig_violin, ax_violin = plt.subplots()
sns.violinplot(x='species', y=selected_violin_feature, data=df, palette='muted', ax=ax_violin)
st.pyplot(fig_violin)

# ---- Footer ---- #
st.write("---")
st.write("© 2024 - Analyse complète des Données Iris • M. Merveille LIGAN")
