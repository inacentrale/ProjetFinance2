import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Clustering KMeans", layout="wide")
st.title(" Clustering des clients avec KMeans")

# ===============================
# Chargement des données
# ===============================
data_path = "data/Loan.csv"

if not os.path.exists(data_path):
    st.error(f"❌ Le fichier '{data_path}' est introuvable.")
    st.stop()

df = pd.read_csv(data_path)

# ===============================
# Clustering
# ===============================
features = ['RiskScore', 'DebtToIncomeRatio', 'NetWorth', 'AnnualIncome']

if not all(col in df.columns for col in features + ['LoanApproved']):
    st.error("⚠️ Certaines colonnes nécessaires sont manquantes dans le fichier.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=3, random_state=42)
df['RiskCluster'] = kmeans.fit_predict(X_scaled)

# ===============================
# Analyse des clusters
# ===============================
st.subheader(" Analyse des clusters")
# Expander 1 : Description des moyennes des variables par cluster
with st.expander(" Moyennes des variables par cluster"):
    st.markdown("""
Ce tableau présente les valeurs moyennes des principales variables financières pour chaque cluster (0, 1 et 2) :

- **RiskScore** : le score de risque moyen dans chaque groupe.
- **DebtToIncomeRatio** : ratio d’endettement moyen (plus c’est élevé, plus le client est endetté par rapport à ses revenus).
- **NetWorth** : le patrimoine moyen des clients du groupe.
- **AnnualIncome** : revenu annuel moyen.
- **LoanApproved** : proportion moyenne de clients ayant eu leur prêt approuvé (par exemple, 0.882 = 88.2 % dans le cluster 2).

 **Interprétation** :  
Ce tableau permet de comparer les profils financiers de chaque cluster.  
Il montre notamment que le **cluster 2** regroupe les clients les plus stables financièrement, tandis que les clusters **0 et 1** affichent un risque plus élevé (et donc de faibles taux d’approbation).
    """)
st.write(df[features + ['LoanApproved', 'RiskCluster']].groupby('RiskCluster').mean())


fig, ax = plt.subplots()
sns.scatterplot(x=df['RiskScore'], y=df['DebtToIncomeRatio'], hue=df['RiskCluster'], palette='Set1', ax=ax)
plt.title("Clusters des clients (RiskScore vs DTI)")
plt.xlabel("RiskScore")
plt.ylabel("Debt to Income Ratio")
st.pyplot(fig)
with st.expander(" Interprétation des clusters"):
    st.markdown("""
 **Cluster 0 (Majoritairement clients à haut risque)**  
- **RiskScore moyen** : 57.3  
- **DTI moyen** : 0.45 (assez élevé → endettement important)  
- **NetWorth** : 51 266 € (modéré)  
- **Revenu annuel** : 46 580 €  
- **Taux d’approbation des prêts** : 5,03 % seulement  
- **Effectif** : 6 205 clients (5 893 refus, 312 approbations)  

 Ce cluster regroupe des clients au niveau de risque élevé avec un taux d’endettement significatif, ce qui explique leur faible taux d’approbation de crédit. Ce groupe devrait faire l'objet d'une politique de crédit très prudente, voire restrictive.

---

 **Cluster 1 (Clients à risque moyen mais peu approuvés)**  
- **RiskScore moyen** : 51.5  
- **DTI moyen** : 0.18 (très bas)  
- **NetWorth** : 55 006 €  
- **Revenu annuel** : 43 652 €  
- **Taux d’approbation** : seulement 4 %  
- **Effectif** : 9 142 clients (8 776 refus, 366 approbations)  

 Ce groupe est intéressant : malgré un DTI faible, ce qui est généralement bon signe, ces clients reçoivent peu d’approbations. Cela pourrait être dû à un score de risque juste moyen, ou à d’autres critères externes non inclus dans l’analyse. Ce groupe mériterait une analyse plus approfondie pour détecter d'éventuels faux négatifs ou pour adapter les politiques d’approbation.

---

 **Cluster 2 (Clients à faible risque – bons candidats au crédit)**  
- **RiskScore moyen** : 40.4 (plus bas, donc mieux)  
- **DTI moyen** : 0.26 (modéré)  
- **NetWorth** : 134 302 € (nettement plus élevé que les autres)  
- **Revenu annuel** : 106 410 € (très élevé)  
- **Taux d’approbation des prêts** : 88,16 %  
- **Effectif** : 4 653 clients (4 102 approbations)  

 Ce groupe représente les meilleurs profils clients pour l’octroi de crédits. Ils disposent d’un patrimoine élevé, d’un bon revenu et d’un score de risque relativement faible. C’est le groupe idéal pour les offres premium (taux préférentiels, montants élevés, fidélisation…).

---

 **Conclusion de l'analyse**  
Le clustering a permis de révéler des profils bien distincts :

- Le **Cluster 2** est clairement le plus solvable.  
- Le **Cluster 0** est à haut risque, surtout à cause d’un DTI élevé.  
- Le **Cluster 1** est paradoxalement peu approuvé malgré un faible DTI, ce qui invite à une revue des critères d’évaluation.

Cette segmentation est très utile pour :  
- Adapter dynamiquement les montants et taux de crédit  
- Prioriser les relances ou campagnes commerciales  
- Appliquer une politique de risque différenciée  

 Cela ouvre également la voie à une personnalisation des produits financiers selon les groupes, et à une amélioration du scoring pour les cas ambigus (notamment dans le cluster 1).
    """)

st.subheader(" Répartition des prêts par cluster")

# Expander 2 : Description de la répartition en pourcentage
with st.expander(" Répartition des décisions par cluster (%)"):
    st.markdown("""
Ce tableau donne la répartition en **pourcentage** des décisions d’approbation ou de refus de prêts dans chaque cluster :

- Pour chaque cluster (0, 1, 2), on voit quel pourcentage des clients a été **refusé (0)** ou **approuvé (1)**.

 **Interprétation** :  
- Les **clusters 0 et 1** ont des taux de refus très élevés (**95–96 %**),  
- Le **cluster 2**, au contraire, regroupe les clients les plus approuvés (**88 %** de taux d’approbation).

Cela met en évidence une **corrélation forte entre le profil du client (déterminé par le cluster) et sa probabilité d’avoir un prêt**.
    """)
st.write(pd.crosstab(df['RiskCluster'], df['LoanApproved'], normalize='index') * 100)

# Expander 3 : Description des effectifs
with st.expander(" Effectifs réels de clients par cluster"):
    st.markdown("""
Il s’agit de la répartition en **effectifs réels** du nombre de clients **approuvés (1)** et **refusés (0)** dans chaque cluster.

 **Interprétation** :  
Ce tableau vous dit exactement **combien de clients sont dans chaque situation** :

- Par exemple, dans le **cluster 2**, **4 102 clients** ont été approuvés et **551 refusés**.  
- Le **cluster 1** est le plus gros groupe avec **9 142 clients** (**8 776 refusés et 366 approuvés**).

Cela permet de **quantifier le volume de chaque segment** et d’envisager des **stratégies ciblées** selon le profil dominant dans chaque groupe.
    """)
st.write(pd.crosstab(df['RiskCluster'], df['LoanApproved']))

# ===============================
# Profils des clusters
# ===============================
cluster_profiles = {}
for cluster in sorted(df['RiskCluster'].unique()):
    cluster_data = df[df['RiskCluster'] == cluster]
    cluster_profiles[cluster] = {
        'taux_approbation': cluster_data['LoanApproved'].mean(),
        'risque_moyen': cluster_data['RiskScore'].mean(),
        'dti_moyen': cluster_data['DebtToIncomeRatio'].mean(),
        'nb_clients': len(cluster_data)
    }
    
    

# ===============================
# Interface utilisateur
# ===============================
st.subheader(" Simulation d'un client")

with st.form("formulaire_client"):
    risk_score = st.slider("Risk Score", 0, 100, 50)
    dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)
    patrimoine = st.number_input("Net Worth (€)", min_value=0, value=20000)
    revenu = st.number_input("Annual Income (€)", min_value=0, value=50000)
    montant_demande = st.number_input("Loan Amount (€)", min_value=0, value=15000)
    submit = st.form_submit_button("Simuler")

if submit:
    client_data = {
        'RiskScore': risk_score,
        'DebtToIncomeRatio': dti,
        'NetWorth': patrimoine,
        'AnnualIncome': revenu,
        'LoanAmount': montant_demande
    }

    X_client = scaler.transform(pd.DataFrame([client_data])[features])
    cluster_predit = kmeans.predict(X_client)[0]
    profil = cluster_profiles[cluster_predit]

    st.write(f" Cluster prédit : {cluster_predit}")
    st.write(f"Taux d'approbation moyen dans ce cluster : {profil['taux_approbation']:.1%}")

    # Simulation de décision
    if profil['taux_approbation'] < 0.1:
        if risk_score <= 20:
            montant_final = montant_demande * 0.5
            taux_interet = 0.08
            decision = "Approuvé avec restrictions"
        else:
            montant_final = 0
            taux_interet = None
            decision = "Refusé"
    elif profil['taux_approbation'] >= 0.6:
        if risk_score <= 40:
            montant_final = montant_demande * 1.1
            taux_interet = 0.035
            decision = "Approuvé - Profil privilégié"
        elif risk_score <= 60:
            montant_final = montant_demande
            taux_interet = 0.045
            decision = "Approuvé"
        else:
            montant_final = montant_demande * 0.6
            taux_interet = 0.08
            decision = "Partiellement approuvé"
    else:
        if risk_score <= 30:
            montant_final = montant_demande
            taux_interet = 0.04
            decision = "Approuvé"
        elif risk_score <= 50:
            montant_final = montant_demande * 0.75
            taux_interet = 0.07
            decision = "Partiellement approuvé"
        elif risk_score <= 70:
            montant_final = montant_demande * 0.4
            taux_interet = 0.12
            decision = "Partiellement approuvé"
        else:
            montant_final = 0
            taux_interet = None
            decision = "Refusé"

    st.success(f" Décision : {decision}")
    st.write(f"Montant accordé : {montant_final:,.0f} €")
    st.write(f"Taux d’intérêt : {taux_interet:.1%}" if taux_interet else "Taux d’intérêt : N/A")

# ===============================
# Sauvegarde
# ===============================
joblib.dump(kmeans, os.path.join("data", "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
cluster_profiles_df = pd.DataFrame.from_dict(cluster_profiles, orient='index')
cluster_profiles_df.index.name = 'RiskCluster'
cluster_profiles_df.to_csv(os.path.join("data", "cluster_profiles.csv"))
