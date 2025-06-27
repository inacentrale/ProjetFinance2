import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Random Forest", layout="wide")
st.title(" Modèle Random Forest pour l'approbation de prêt")

# Chargement des données
df = pd.read_csv("data/Loan.csv")

# Sélection des variables explicatives et de la cible
X = df[['Age', 'AnnualIncome', 'RiskScore', 'LoanAmount', 'LoanDuration',
        'InterestRate', 'BaseInterestRate', 'MonthlyLoanPayment',
        'NetWorth', 'PreviousLoanDefaults', 'DebtToIncomeRatio']]
y = df['LoanApproved']

# Division du dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialisation du modèle
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=3,
    min_samples_leaf=10,
    random_state=42
)

# Validation croisée
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
st.metric(" Accuracy (CV 5 folds)", f"{scores.mean():.2%}")
with st.expander(" Interprétation de la validation croisée "):
    st.write("""
    Le modèle Random Forest a été évalué à l’aide d’une validation croisée à 5 plis (**5-fold cross-validation**).

     **Résultat obtenu :** une accuracy moyenne de **96.84 %** sur les 5 plis.

     **Interprétation :**
    - Le modèle prédit correctement l’approbation ou le refus de prêt dans environ **97 cas sur 100**, ce qui est **très performant**.
    - Cela montre que le Random Forest est capable de **capturer des relations complexes** entre les variables sans trop surapprendre.
    - Ce bon score de validation croisée indique que le modèle est **stable** et **généralise bien** sur différents échantillons de données.

     **Comparaison ** :
    - Par rapport à la régression logistique (~97.85 %), le modèle Random Forest est **légèrement moins précis**, mais il pourrait mieux gérer les non-linéarités et les interactions entre variables.
    - Nous observerons aussi les métriques comme le **recall**, la **précision** ou encore l’**importance des variables** pour un diagnostic plus complet.

    
    """)


# Entraînement
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Rapport de classification
st.subheader(" Rapport de classification")
st.text(classification_report(y_test, y_pred))

#  Matrice de confusion avec labels clairs
st.subheader(" Matrice de confusion")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Refusé", "Approuvé"],
            yticklabels=["Refusé", "Approuvé"],
            ax=ax)
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
st.pyplot(fig)

with st.expander(" Interprétation des performances du modèle Random Forest"):
    st.write("""
Le modèle **Random Forest** présente de très bonnes performances globales sur l'ensemble de test :

- **Accuracy globale** : `97%`, ce qui indique que le modèle fait très peu d’erreurs.
- **F1-score** élevé pour les deux classes :
  - `Refusé (classe 0)` : F1-score de **0.98**, reflétant une excellente capacité à identifier les clients à qui le prêt doit être refusé.
  - `Approuvé (classe 1)` : F1-score de **0.94**, montrant que le modèle gère aussi bien les clients à approuver, mais avec une légère marge d’erreur.

###  Analyse de la matrice de confusion :

-  **Vrais négatifs (4455)** : Nombre de clients refusés correctement par le modèle.
-  **Vrais positifs (1369)** : Clients approuvés correctement.
-  **Faux positifs (67)** : Le modèle a approuvé 67 clients qui auraient dû être refusés (risque potentiel de défaut).
-  **Faux négatifs (109)** : Le modèle a refusé 109 clients qui auraient pu être approuvés (manque à gagner potentiel).

###  Conclusion :

Le modèle est **particulièrement performant** pour détecter les clients à qui il faut refuser un crédit, ce qui est essentiel pour limiter le risque. Toutefois, **quelques clients valables sont encore refusés** (faux négatifs), ce qui laisse une **marge d’amélioration pour mieux capter les bons profils** sans augmenter le risque.
""")
    





#  Simulation client
st.subheader(" Simulation : Tester un nouveau client")

with st.form("form_client_rf"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=35)
        annual_income = st.number_input("Revenu annuel", min_value=0.0, value=30000.0, step=100.0)
        risk_score = st.number_input("Risk Score", min_value=0, max_value=100, value=50)
        loan_amount = st.number_input("Montant du prêt", min_value=0.0, value=2000.0)
        loan_duration = st.number_input("Durée du prêt (mois)", min_value=1, value=12)
        interest_rate = st.number_input("Taux d’intérêt (%)", min_value=0.0, value=5.0, step=0.1)

    with col2:
        base_rate = st.number_input("Taux de base", min_value=0.0, value=3.0, step=0.1)
        monthly_payment = st.number_input("Paiement mensuel", min_value=0.0, value=200.0)
        net_worth = st.number_input("Patrimoine net", min_value=0.0, value=10000.0)
        previous_defaults = st.number_input("Nombre d’incidents précédents", min_value=0, value=0)
        debt_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, value=0.5, step=0.01)

    submitted = st.form_submit_button("Prédire")

    if submitted:
        client_data = pd.DataFrame([{
            'Age': age,
            'AnnualIncome': annual_income,
            'RiskScore': risk_score,
            'LoanAmount': loan_amount,
            'LoanDuration': loan_duration,
            'InterestRate': interest_rate,
            'BaseInterestRate': base_rate,
            'MonthlyLoanPayment': monthly_payment,
            'NetWorth': net_worth,
            'PreviousLoanDefaults': previous_defaults,
            'DebtToIncomeRatio': debt_ratio
        }])

        prediction = model.predict(client_data)[0]
        proba = model.predict_proba(client_data)[0][prediction]

        if prediction == 1:
            st.success(f"✅ Crédit **APPROUVÉ** avec une probabilité de {proba:.2%}")
            with st.expander("ℹ️ Pourquoi cette approbation ?", expanded=False):
                justifications = []
                if risk_score >= 60:
                    justifications.append("- Le **RiskScore** est élevé (≥ 60), indiquant un bon profil de risque.")
                if debt_ratio <= 0.4:
                    justifications.append("- Le **Debt to Income Ratio** est raisonnable (≤ 0.4), le client n’est pas trop endetté.")
                if previous_defaults == 0:
                    justifications.append("- Aucun **incident de crédit précédent** signalé.")
                if not justifications:
                    justifications.append("- Profil globalement acceptable pour l’approbation.")

                st.write("\n".join(justifications))

        else:
            st.error(f"❌ Crédit **REFUSÉ** avec une probabilité de {proba:.2%}")
            with st.expander("ℹ️ Pourquoi ce refus ?", expanded=False):
                justifications = []
                if risk_score < 40:
                    justifications.append("- Le **RiskScore** est faible (< 40), indiquant un risque élevé.")
                if debt_ratio > 0.6:
                    justifications.append("- Le **niveau d’endettement** est élevé (> 0.6).")
                if previous_defaults > 0:
                    justifications.append(f"- Historique de crédit : {previous_defaults} **incident(s)** de crédit précédent.")
                if not justifications:
                    justifications.append("- Le modèle considère ce profil comme trop risqué pour une approbation.")
                    
                st.write("\n".join(justifications))
