"""Projet 2 - Prédiction avec le modèle entraîné."""
import os

import joblib
import pandas as pd

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "model.joblib")
_modele = None


def predict(df: pd.DataFrame) -> list:
    """df : mêmes colonnes que flux_train.csv, SANS 'attaque'.
    Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne."""
    global _modele
    if _modele is None:
        _modele = joblib.load(_MODEL_PATH)
    df = df.drop(columns=["attaque"], errors="ignore")
    return [int(p) for p in _modele.predict(df)]


if __name__ == "__main__":
    donnees = pd.read_csv(os.path.join(os.path.dirname(_MODEL_PATH), "..", "data", "flux_train.csv"))
    print(predict(donnees.head(10)))
