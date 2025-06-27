import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Régression Logistique", layout="wide")
st.title("🔍 Analyse avec Régression Logistique")
st.write("Ce module applique une régression logistique pour prédire l'approbation d'un prêt.")

# Chargement des données
df = pd.read_csv("data/Loan.csv")

# Sélection des variables
df_clean = df[['RiskScore', 'DebtToIncomeRatio', 'NetWorth', 'AnnualIncome', 'LoanApproved']].dropna()

X = df_clean[['RiskScore', 'DebtToIncomeRatio', 'NetWorth', 'AnnualIncome']]
y = df_clean['LoanApproved']

# Split identique
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Création du pipeline
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, C=0.01, penalty='l2', class_weight='balanced'))
])

# Validation croisée
logreg_scores = cross_val_score(logreg_pipeline, X, y, cv=5, scoring='accuracy')
st.metric("🎯 Accuracy (CV 5 folds)", f"{logreg_scores.mean():.2%}")
with st.expander(" Interprétation de la validation croisée "):
    st.write("""
    Le modèle de régression logistique a été évalué à l'aide d'une validation croisée à 5 plis (**5-fold cross-validation**).

     **Résultat obtenu :** une accuracy moyenne de **97.85 %** sur les 5 plis.

     **Interprétation :**
    - Cela signifie que le modèle parvient à prédire correctement l'approbation ou le refus de prêt dans environ **98 cas sur 100** sur des sous-ensembles différents des données.
    - Une telle performance indique que le modèle est **robuste**, **généralise bien** et ne dépend pas excessivement d'un seul échantillon de données.
    - La validation croisée permet également de réduire les risques de surapprentissage (*overfitting*) en s'assurant que le modèle est testé sur différentes portions du jeu de données.
    
    
    """)


# Entraînement et prédictions
logreg_pipeline.fit(X_train, y_train)
y_pred_logreg = logreg_pipeline.predict(X_test)

# ========================
# 📉 Matrice de confusion
# ========================
st.subheader("📉 Matrice de confusion")
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
fig, ax = plt.subplots()
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Refusé', 'Approuvé'], yticklabels=['Refusé', 'Approuvé'], ax=ax)
plt.xlabel("Prédiction")
plt.ylabel("Réel")
st.pyplot(fig)



st.write("""



- **Vrai positif (VP = 999)** : le modèle a correctement prédit l’approbation du prêt pour 999 clients.
- **Faux négatif (FN = 18)** : 18 clients dont le prêt devait être approuvé ont été à tort classés comme refusés.
- **Vrai négatif (VN = 2925)** : 2925 refus ont été correctement identifiés par le modèle.
- **Faux positif (FP = 58)** : 58 clients ont été approuvés à tort, alors qu’ils auraient dû être refusés.




""")


# ========================
# 📄 Rapport de classification
# ========================
st.subheader("📄 Rapport de classification")
# Générer le rapport sous forme de dictionnaire
report_dict = classification_report(y_test, y_pred_logreg, target_names=["Refusé", "Approuvé"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
# Afficher le rapport joliment dans Streamlit
table_style = report_df.style.format("{:.2f}").set_properties(**{"text-align": "center"})
st.dataframe(table_style, use_container_width=True)



st.write("""


####  **Interprétation du rapport de classification**

- **Accuracy globale** : **98%**  
  → Le modèle est **très performant** avec seulement **2% d'erreurs globales**.

#####  Classe "Refusé"
- **Précision = 0.99** → Parmi tous les clients prévus comme refusés, 99% l’étaient réellement.
- **Recall = 0.98** → Le modèle identifie correctement 98% des refus réels.
- **F1-score = 0.99** → Très bon équilibre entre précision et rappel.

#####  Classe "Approuvé"
- **Précision = 0.95** → Parmi les clients prédits comme approuvés, 95% ont réellement eu leur prêt accordé.
- **Recall = 0.98** → Le modèle capture 98% des clients à approuver.
- **F1-score = 0.96** → Très bon compromis entre précision et rappel, malgré un léger déséquilibre.



#####  **Ce que cela signifie :**
- Le modèle est **très fiable** pour **prédire les refus**, avec très peu de faux positifs.
- Il est aussi **bon pour approuver** les bons dossiers, même s’il a un **petit taux d’erreurs d’approbation (FP)** qui pourrait être coûteux si les clients à risque obtiennent un prêt par erreur.
- Les **faux négatifs** sont très faibles (seulement 18 sur 1017), ce qui est **positif**, car peu de bons clients sont injustement refusés.



####  **Conclusion**
Ce modèle de régression logistique est bien calibré, **précis et équilibré**, adapté à un usage en pré-qualification ou en support aux décisions d’octroi de prêts.  
Il conviendra toutefois de renforcer la vérification humaine sur les cas limites, surtout autour des **clients approuvés à faible probabilité**, pour **minimiser les risques de défaut**.
""")



# ========================
# 🧪 Tester un nouveau client
# ========================
st.subheader(" Simulation : Tester un nouveau client")

with st.form("formulaire_client"):
    col1, col2 = st.columns(2)

    with col1:
        risk_score = st.number_input("Risk Score", min_value=0, max_value=100, value=50)
        debt_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=5.0, value=0.5, step=0.01)

    with col2:
        net_worth = st.number_input("Net Worth", min_value=0.0, value=10000.0, step=100.0)
        annual_income = st.number_input("Annual Income", min_value=0.0, value=30000.0, step=100.0)

    submitted = st.form_submit_button("Prédire")

    if submitted:
        # Création d'un DataFrame pour le client simulé
        client_data = pd.DataFrame([{
            'RiskScore': risk_score,
            'DebtToIncomeRatio': debt_ratio,
            'NetWorth': net_worth,
            'AnnualIncome': annual_income
        }])

        # Prédiction
        prediction = logreg_pipeline.predict(client_data)[0]
        proba = logreg_pipeline.predict_proba(client_data)[0][prediction]

        # Affichage du résultat
        if prediction == 1:
            st.success(f"✅ Crédit **APPROUVÉ** avec une probabilité de {proba:.2%}")
            st.markdown("###  Justification possible :")
            if risk_score >= 60 and annual_income >= 40000 and debt_ratio <= 0.5:
                st.write("- Le client a un **bon score de risque** et un **revenu stable**.")
                st.write("- Son **niveau d'endettement est raisonnable**, ce qui le rend solvable.")
            elif net_worth > 20000:
                st.write("- Bien que le revenu soit moyen, le client a un **patrimoine élevé** (Net Worth) pouvant rassurer sur sa capacité de remboursement.")
            else:
                st.write("- Les caractéristiques financières du client sont **suffisamment solides** selon le modèle.")
        
        else:
            st.error(f"❌ Crédit **REFUSÉ** avec une probabilité de {proba:.2%}")
            st.markdown("###  Justification possible :")
            if risk_score <= 30:
                st.write("- Le **RiskScore est trop faible**, ce qui indique un profil risqué.")
            if debt_ratio > 0.6:
                st.write("- Le client a un **niveau d'endettement élevé**.")
            if annual_income < 20000:
                st.write("- Le **revenu annuel est trop faible** pour garantir un remboursement sûr.")
            if net_worth < 5000:
                st.write("- Le **patrimoine du client est insuffisant**, ce qui augmente le risque.")
            if risk_score > 30 and annual_income >= 20000 and net_worth >= 5000:
                st.write("- Bien que les critères soient acceptables, **le modèle a identifié un léger risque** justifiant un refus.")
