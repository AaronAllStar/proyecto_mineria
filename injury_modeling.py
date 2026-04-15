import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Configuración de estilo Dinosaur-like
sns.set_theme(style="whitegrid")

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Limpieza básica
    # Extraer el número de '43 days'
    if df['Days'].dtype == 'object':
        df['Days'] = df['Days'].str.replace(' days', '', regex=False)
    
    # Convertir 'Days' a numérico por si acaso
    df['Days'] = pd.to_numeric(df['Days'], errors='coerce')
    df = df.dropna(subset=['Days', 'player_age'])
    
    # 2. Ingeniería de Características (Feature Engineering)
    # Agrupar lesiones similares (opcional pero recomendado si hay muchas categorías)
    # Por ahora usaremos 'Injury' directamente
    
    return df

def analytical_insights(df):
    print("\n--- ANALYTICAL INSIGHTS ---")
    
    # Equipos con más lesiones históricas
    top_injury_clubs = df.groupby('club')['Injury'].count().sort_values(ascending=False).head(10)
    print("\nTop 10 Clubes con mayor tendencia a lesiones:")
    print(top_injury_clubs)
    
    # Relación Edad vs Días de recuperación
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='player_age', y='Days', alpha=0.3, color='teal')
    plt.title('Relación Edad del Jugador vs Días de Recuperación')
    plt.savefig('age_vs_days.png')
    
    # Promedio de días de recuperación por posición
    pos_recovery = df.groupby('player_position')['Days'].mean().sort_values(ascending=False)
    print("\nPromedio de días de recuperación por posición:")
    print(pos_recovery)
    
    return top_injury_clubs

def predictive_modeling(df):
    print("\n--- PREDICTIVE MODELING ---")
    
    # Selección de variables para el modelo
    # X: player_age, player_position, club, Injury (tipo)
    # y: Days (duración)
    
    features = ['player_age', 'player_position', 'club', 'Injury']
    X = df[features].copy()
    y = df['Days']
    
    # Encode categorical variables
    encoders = {}
    for col in ['player_position', 'club', 'Injury']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model: Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluación
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Absolute Error: {mae:.2f} días")
    print(f"R2 Score: {r2:.4f}")
    
    # Importancia de características
    importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    importances = importances.sort_values(by='importance', ascending=False)
    print("\nImportancia de variables en la predicción:")
    print(importances)
    
    return model, encoders

if __name__ == "__main__":
    file_path = "full_dataset_thesis - 1.csv"
    data = load_and_preprocess(file_path)
    
    # 1. Análisis
    clubs_tendency = analytical_insights(data)
    
    # 2. Modelo
    model, encoders = predictive_modeling(data)
    
    print("\n[Dinosaur AI] Proceso de modelado completado con éxito.")
