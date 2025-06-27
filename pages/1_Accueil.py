import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# for Streamlit rebuild

st.set_page_config(page_title="Accueil", layout="wide")
st.title("📊 Tableau de bord - Analyse des données")

# Chargement des données
df = pd.read_csv("data/Loan.csv")

# =============================
# 🗃️ Aperçu du dataset
# =============================
with st.expander("🔍 Aperçu et description du dataset"):
    st.write("Ici nous affichons les premières lignes du dataset (Loan.csv) utilisé dans le projet. ")
    st.write(" Cela permet de visualiser les colonnes principales, comprendre la structure des données et identifier les types de variables (numériques, catégorielles...). ")
    st.dataframe(df.head())
    st.subheader("📃 Description statistique")
    st.write("Cette section fournit des statistiques descriptives pour chaque colonne numérique du dataset, telles que la moyenne, l'écart-type, les valeurs minimales et maximales...")
    st.write(df.describe())

# =============================
# 🔗 Matrice de corrélation
# =============================
with st.expander("📌 Matrice de corrélation"):
    st.write("Cette carte montre la corrélation entre les variables numériques. Plus la couleur est intense, plus la corrélation est forte.")
    st.write("Elle a permis :\n"
             "- D'identifier les relations linéaires entre les variables (ex : LoanAmount et MonthlyPayment.)\n"
             "- De détecter d’éventuelles redondances ou des variables très influentes (comme RiskScore ou DebtToIncomeRatio.)\n"
             "- De guider la sélection des variables pour les modèles prédictifs.")
    correlation_matrix = df.corr(numeric_only=True)
    n = len(correlation_matrix)
    fig_width = max(12, n * 0.6)
    fig_height = max(8, n * 0.5)
    fig_corr, ax_corr = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax_corr)
    ax_corr.set_title('Carte des corrélations', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig_corr)

# =============================
# 📈 Distribution du RiskScore
# =============================
with st.expander("📊 Distribution du RiskScore"):
    st.write("Cet histogramme montre la distribution des scores de risque parmi les clients.\n"
             "Il  permet de comprendre comment se répartit le niveau de risque des clients. On observe notamment deux pics : au niveau du RiskScore égale à 40 et des scores compris entre 50  et 60.")
    st.write("""
### ℹ️ Interprétation de la distribution du RiskScore


-  **La majorité des clients** ont un RiskScore entre **45 et 55**, ce qui indique un **niveau de risque modéré**.
-  La distribution n’est pas parfaitement symétrique : il y a **plus de clients légèrement en dessous de 50** que très au-dessus.
-  Les clients avec un score supérieur à **70** sont **rares** et représentent des **profils très risqués**.
- De même, les profils très sûrs (RiskScore < 35) sont peu fréquents.

**Conclusion :**  
La population des clients est majoritairement composée de profils à **risque moyen**, avec peu de cas extrêmes. Cela peut aider à mieux cibler les décisions d'approbation de crédit ou de segmentation des offres.
""")

    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    sns.histplot(df['RiskScore'], kde=True, bins=30, ax=ax_dist)
    ax_dist.set_title("Distribution du RiskScore")
    ax_dist.set_xlabel("RiskScore")
    ax_dist.set_ylabel("Nombre de clients")
    ax_dist.grid(True)
    st.pyplot(fig_dist)

# =============================
# 🧩 Clustering avec KMeans
# =============================
with st.expander("🧠 Analyse de clusters (méthode du coude et visualisation)"):
    

    clustering_features = df[['RiskScore', 'DebtToIncomeRatio', 'NetWorth', 'AnnualIncome', 'LoanApproved']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clustering_features)

    # Méthode du coude
    st.write("""
    ### 🔍 Méthode du coude 

    La méthode du coude permet de **déterminer le nombre optimal de clusters** pour l’algorithme KMeans.


    - Dans notre cas, on observe un coude autour de **k = 3**, ce qui suggère que **3 groupes de clients distincts** permettent une bonne segmentation.
    - Ce choix est confirmé par l’analyse des groupes (`Cluster 0`, `Cluster 1`, `Cluster 2`) qui ont des comportements différents vis-à-vis de l’approbation de prêt.

    ✅ Ce nombre est donc utilisé pour la suite de l’analyse de clustering.
    """)
    inertias = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
    ax_elbow.plot(range(1, 10), inertias, marker='o')
    ax_elbow.set_xlabel('Nombre de clusters')
    ax_elbow.set_ylabel('Inertie (distortion)')
    ax_elbow.set_title('Méthode du coude pour KMeans')
    ax_elbow.grid(True)
    st.pyplot(fig_elbow)

    # Visualisation des clusters (avec 3 clusters pour l'exemple)
    st.write("""
###  Interprétation détaillée du clustering KMeans

L’algorithme KMeans a segmenté les clients en **3 clusters** (groupes) selon leurs caractéristiques financières (`RiskScore`, `NetWorth`, `DTI`, `AnnualIncome`, `LoanApproved`.).  




-  **Les clusters 0 & 1** regroupent des clients avec **profils financiers défavorables** (fort endettement, faible revenu, mauvais score de risque, etc.).  
  → Très peu de prêts sont accordés, ce sont donc des **groupes à surveiller de près**.

-  **Le cluster 2** contient la grande majorité des clients dont le prêt est approuvé.  
  → Ces clients représentent un **profil type "solvable" ou à faible risque**, idéal pour des offres de crédit ou de fidélisation.

####  Utilisation recommandée :
- Adapter les **stratégies marketing** ou les **offres** selon le cluster.
- Renforcer la **politique de scoring automatique** à partir des caractéristiques les plus discriminantes (ex : DTI, NetWorth).
- Évaluer le risque global du portefeuille client par segmentation.
""")

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_clustered = clustering_features.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(X_scaled)

    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df_clustered, x='RiskScore', y='DebtToIncomeRatio',
        hue='Cluster', palette='Set1', s=60, ax=ax_cluster
    )
    ax_cluster.set_title("Visualisation des clusters : RiskScore vs DebtToIncomeRatio")
    ax_cluster.set_xlabel("RiskScore")
    ax_cluster.set_ylabel("DebtToIncomeRatio")
    ax_cluster.grid(True)
    st.pyplot(fig_cluster)
    
    
    
    

