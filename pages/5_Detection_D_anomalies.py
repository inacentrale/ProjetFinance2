import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Détection d'anomalies", layout="wide")
st.title("🚨 Détection d’anomalies avec Isolation Forest")

st.markdown("""
### 🎯 Objectif général de l'algorithme

La détection des clients suspects (**potentiellement à risque ou frauduleux**) en se basant sur leurs **caractéristiques financières** 
(Age, AnnualIncome, CreditScore, Experience, LoanAmount, LoanDuration, MonthlyIncome, NetWorth, InterestRate, 'MonthlyLoanPayment',
    TotalDebtToIncomeRatio, RiskScore), grâce à l’algorithme d’**Isolation Forest**, et visualiser les résultats dans une **interface web interactive**.
""")

st.markdown("""
Cette page utilise **Isolation Forest** pour détecter des clients potentiellement suspects à partir de plusieurs variables financières.
""")

with st.expander("📘 Détails sur le modèle Isolation Forest", expanded=False):
    st.markdown("""
    - 🔍 **Nous avons entraîné un modèle d'Isolation Forest**, qui isole les points atypiques dans l’espace de données.
    -  **Résultat** :
        - `anomaly_labels = -1` → **Client suspect**
        - `anomaly_labels = 1` → **Client normal**
    """)


# Chargement des données
df = pd.read_csv("data/Loan.csv")

# Variables utilisées pour la détection
features = [
    'Age', 'AnnualIncome', 'CreditScore', 'Experience',
    'LoanAmount', 'LoanDuration', 'MonthlyIncome',
    'NetWorth', 'InterestRate', 'MonthlyLoanPayment',
    'TotalDebtToIncomeRatio', 'RiskScore'
]

# Nettoyage et normalisation
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modèle Isolation Forest
iso_forest = IsolationForest(contamination=0.03, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)

# Ajout des colonnes au DataFrame
df_anomaly = X.copy()
df_anomaly['AnomalyScore'] = iso_forest.decision_function(X_scaled)
df_anomaly['Anomaly'] = anomaly_labels

# ==========================
# 📊 Affichage du scatterplot
# ==========================
st.subheader("📈 Visualisation des anomalies (RiskScore vs DTI)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df_anomaly,
    x='RiskScore',
    y='TotalDebtToIncomeRatio',
    hue='Anomaly',
    palette={1: "blue", -1: "red"},
    ax=ax
)
plt.title("Détection d’anomalies : RiskScore vs Total DTI")
plt.xlabel("RiskScore")
plt.ylabel("Total Debt to Income Ratio")
plt.legend(title="Anomalie", labels=["Normal", "Anomalie"])
st.pyplot(fig)

with st.expander("ℹ️ Explication de la visualisation", expanded=False):
    st.markdown("""
    - 📍 **Chaque point représente un client**  
    - 🧮 **Axe X** : `RiskScore` (niveau de risque estimé du client)  
    - 📊 **Axe Y** : `TotalDebtToIncomeRatio` (niveau d’endettement total du client)  
    - 🔵 Les **clients normaux** sont représentés en **bleu**  
    - 🔴 Les **clients suspects** (anomalies détectées) sont en **rouge**
    """)

# ==========================
# 🔎 Top clients suspects
# ==========================
st.subheader("🔎 Top 10 clients suspects")
clients_suspects = df_anomaly[df_anomaly['Anomaly'] == -1]
clients_suspects = clients_suspects.sort_values(by='AnomalyScore')
st.dataframe(clients_suspects.head(10).style.format({"AnomalyScore": "{:.4f}"}))

# (Optionnel) Export
with st.expander("📥 Télécharger les résultats complets"):
    st.download_button(
        label="Télécharger les anomalies (CSV)",
        data=clients_suspects.to_csv(index=False).encode('utf-8'),
        file_name="clients_suspects.csv",
        mime='text/csv'
    )

# ==========================
# 🧪 Simulation d’un nouveau client
# ==========================
st.subheader("🧪 Simulation : tester un nouveau client")

with st.form("simulation_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=35)
        experience = st.number_input("Années d’expérience", min_value=0, max_value=80, value=10)
        credit_score = st.number_input("Credit Score", min_value=0.0, max_value=1.0, step=0.01, value=0.6)
        loan_amount = st.number_input("Montant du prêt", min_value=0.0, value=5000.0)

    with col2:
        loan_duration = st.number_input("Durée du prêt (mois)", min_value=1, value=24)
        interest_rate = st.number_input("Taux d'intérêt (%)", min_value=0.0, max_value=100.0, value=5.0)
        net_worth = st.number_input("Patrimoine net", min_value=0.0, value=10000.0)
        monthly_income = st.number_input("Revenu mensuel", min_value=0.0, value=3000.0)

    with col3:
        annual_income = st.number_input("Revenu annuel", min_value=0.0, value=36000.0)
        monthly_payment = st.number_input("Paiement mensuel du prêt", min_value=0.0, value=200.0)
        dti = st.number_input("Total Debt to Income Ratio", min_value=0.0, max_value=1.0, step=0.01, value=0.3)
        risk_score = st.number_input("Risk Score", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    submitted = st.form_submit_button("Analyser le client")

    if submitted:
        new_data = pd.DataFrame([[
            age, annual_income, credit_score, experience,
            loan_amount, loan_duration, monthly_income,
            net_worth, interest_rate, monthly_payment,
            dti, risk_score
        ]], columns=features)

        # Normalisation
        new_data_scaled = scaler.transform(new_data)

        # Prédiction
        prediction = iso_forest.predict(new_data_scaled)[0]
        score = iso_forest.decision_function(new_data_scaled)[0]

        if prediction == -1:
            st.error(f"🚨 Ce client est détecté comme **suspect** (anomalie).")
        else:
            st.success("✅ Ce client est considéré comme **normal**.")

        st.markdown(f"**Score d'anomalie** : `{score:.4f}` (plus le score est négatif, plus l’anomalie est forte)")
