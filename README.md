# ImmoBird - Projet d'Estimation Immobilière

Ce projet est une application web d'estimation immobilière qui utilise des modèles de machine learning pour prédire les prix des biens immobiliers.

## Structure du Projet

```
ImmoBird/
├── data/               # Données brutes et prétraitées
│   ├── house_pred.csv
│   └── house_pred_for_ml.csv
├── src/               # Scripts Python
│   ├── app.py
│   ├── generate_data.py
│   ├── data_processing.py
│   └── model_training.py
├── static/            # Fichiers statiques
│   ├── styles.css
│   └── ImmoBird_logo.png
├── templates/         # Templates HTML
│   └── index.html
├── models/            # Modèles sauvegardés
│   └── immo_model.joblib
├── requirements.txt   # Dépendances Python
└── README.md         # Documentation
```

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/MacOS
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Générer les données :
```bash
python src/generate_data.py
```

2. Lancer l'application :
```bash
python src/app.py
```

## Fonctionnalités

- Génération de données immobilières synthétiques
- Prétraitement des données
- Entraînement de modèles de prédiction
- Interface web pour l'estimation des prix 