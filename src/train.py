"""Projet 2 - Entraînement du détecteur d'attaques réseau.

Lancement : python src/train.py   (depuis la racine du dépôt)
"""
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA = "data/flux_train.csv"
MODEL_PATH = "models/model.joblib"
CIBLE = "attaque"
NUM = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
       "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
CAT = ["protocol"]


def construire(modele):
    """Pipeline complet : imputation + échelle + encodage + modèle."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", modele)])


def main():
    df = pd.read_csv(DATA)

    # 1. Exploration rapide
    print("Taille :", df.shape)
    print("Valeurs manquantes :\n", df.isna().sum()[df.isna().sum() > 0])
    print("Répartition de la cible :\n", df[CIBLE].value_counts(normalize=True).round(3))

    X, y = df[NUM + CAT], df[CIBLE]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # 2. Modèles à comparer
    modeles = {
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Arbre de décision": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    }

    scores = {}
    for nom, m in modeles.items():
        pipe = construire(m).fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        scores[nom] = f1_score(y_val, pred)
        print(f"\n===== {nom} =====")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"], digits=3))
        print("Matrice de confusion [[VN FP] [FN VP]] :\n", confusion_matrix(y_val, pred))

    # 3. Meilleur modèle -> ré-entraînement sur toutes les données
    meilleur = max(scores, key=scores.get)
    print(f"\nModèle retenu : {meilleur} (F1 attaque = {scores[meilleur]:.3f})")
    final = construire(modeles[meilleur]).fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print("Modèle enregistré dans", MODEL_PATH)


if __name__ == "__main__":
    main()
