# Projet 2 — Détecter des attaques réseau avec le Machine Learning

## Résultats

**Nom / Prénom :** DIONE PAPE DJIBRIL

Évaluation sur un jeu de validation de 600 flux (20 % des données, découpage stratifié, `random_state=42`).

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) |
|---|---|---|---|
| Régression logistique | 0,920 | 0,898 | 0,943 |
| Arbre de décision (profondeur 6) | 0,953 | 0,964 | 0,942 |
| **Forêt aléatoire (200 arbres)** | **0,982** | **0,976** | **0,988** |

**Modèle retenu et pourquoi :** la forêt aléatoire, qui a le meilleur F1 sur la classe « attaque » (0,982) et le meilleur rappel. Les attaques ont des comportements différents (scan, force brute, DoS) que la régression logistique, un modèle linéaire, sépare moins bien. La forêt les capture mieux, sans surapprentissage visible. Elle a ensuite été ré-entraînée sur les 3000 flux et enregistrée dans `models/model.joblib`.

**Analyse des erreurs (faux négatifs / faux positifs) :** matrice de confusion de la forêt sur la validation :

| | Prédit normal | Prédit attaque |
|---|---|---|
| Réel normal | 431 | 2 (faux positifs) |
| Réel attaque | 4 (faux négatifs) | 163 |

Il y a **4 attaques ratées** sur 167 et 2 fausses alertes. Le faux négatif est le pire cas en sécurité : une attaque non détectée passe sans alerte et peut compromettre le réseau, alors qu'un faux positif coûte seulement une vérification inutile. Comme `class_weight="balanced"` pénalise déjà davantage les attaques manquées, on pourrait aussi baisser le seuil de décision pour gagner encore en rappel, au prix de plus de fausses alertes.

## Lancer le projet
```bash
pip install -r requirements.txt
python src/train.py     # entraîne, compare les modèles, écrit models/model.joblib
python src/predict.py   # petit exemple de prédiction
```

## Données
`data/flux_train.csv` : 3000 flux, 28 % d'attaques, 90 valeurs manquantes sur `duration` (imputées par la médiane). Prétraitement dans un `Pipeline` : imputation, mise à l'échelle, encodage One-Hot de `protocol`.
