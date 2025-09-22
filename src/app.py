from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import time

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__,
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)

print("Chargement du modèle...")
# Chargement du modèle et des scalers
model = joblib.load(os.path.join(BASE_DIR, 'models', 'immo_model.joblib'))
print("Modèle chargé avec succès !")
print("Type du modèle:", type(model))
# print("Paramètres du modèle:", model.get_params())

# Chargement des données d'entraînement pour obtenir les paramètres de standardisation
df_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'house_pred_for_ml.csv'))
df_raw = pd.read_csv(os.path.join(BASE_DIR, 'data', 'house_pred.csv'))  # Données brutes pour l'encodage

# Initialisation des préprocesseurs
scaler = StandardScaler()
scaler.fit(df_raw[['Area', 'YearBuilt']])

# Initialisation du OneHotEncoder avec les mêmes catégories que l'entraînement
ohe = OneHotEncoder(sparse_output=False, drop='first')
ohe.fit(df_raw[['Location', 'Garage', 'Condition']])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory(STATIC_DIR, 'styles.css')

@app.route('/ImmoBird_logo.png')
def logo():
    return send_from_directory(STATIC_DIR, 'ImmoBird_logo.png')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        print("\n=== Début de la prédiction ===")
        
        # Récupération des données du formulaire
        data = request.get_json()
        print("Données reçues:", data)
        
        # Création d'un DataFrame avec les données numériques
        features = pd.DataFrame({
            'Area': [float(data['surface'])],
            'YearBuilt': [int(data['year'])]
        })
        print("Features numériques créées:", features)
        
        # Standardisation des variables numériques
        features[['Area', 'YearBuilt']] = scaler.transform(features[['Area', 'YearBuilt']])
        print("Features numériques standardisées:", features)
        
        # Création d'un DataFrame avec les données catégorielles
        cat_data = pd.DataFrame({
            'Location': [data['location']],
            'Garage': [data['garage']],
            'Condition': [data['condition']]
        })
        print("Features catégorielles créées:", cat_data)
        
        # One Hot Encoding des variables catégorielles
        cat_encoded = ohe.transform(cat_data)
        print("Features catégorielles encodées:", cat_encoded)
        
        # Ajout des variables encodées dans le même ordre que l'entraînement
        features['Location_Lyon'] = cat_encoded[0][0]
        features['Location_Marseille'] = cat_encoded[0][1]
        features['Location_Paris'] = cat_encoded[0][2]
        features['Garage_Oui'] = cat_encoded[0][3]
        features['Condition_Fair'] = cat_encoded[0][4]
        features['Condition_Good'] = cat_encoded[0][5]
        features['Condition_Poor'] = cat_encoded[0][6]
        
        # Vérification de l'ordre des colonnes
        expected_columns = ['Area', 'YearBuilt', 'Location_Lyon', 'Location_Marseille', 
                          'Location_Paris', 'Garage_Oui', 'Condition_Fair', 
                          'Condition_Good', 'Condition_Poor']
        features = features[expected_columns]
        
        print("Features finales:", features)
        
        # Simulation d'un temps de traitement
        print("Simulation d'un temps de traitement de 10 secondes...")
        time.sleep(10)
        
        # Prédiction
        print("Appel du modèle pour la prédiction...")
        prediction = model.predict(features)[0]
        print("Prédiction obtenue:", prediction)
        
        end_time = time.time()
        print(f"Temps total de prédiction: {end_time - start_time:.3f} secondes")
        print("=== Fin de la prédiction ===\n")
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'formatted_prediction': f"{prediction:,.2f} €"
        })
        
    except Exception as e:
        print("Erreur:", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001) 