from flask import Flask, json, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
df = pd.read_csv("Careers.csv").dropna()
#df["contenido"] = df["estado"].fillna("") + " " + df["nombre_y"].fillna("")

# Vectorizar texto
#vectorizer = TfidfVectorizer()
#tfidf_matrix = vectorizer.fit_transform(df["contenido"])

# Crear app Flask
app = Flask(__name__)

@app.route("/recomendar-bigfive", methods=["POST"])
def recomendar_bigfive():
    """Recomienda universidades basado en perfil Big Five"""
    data = request.get_json()
    
    # Esperar un objeto con los traits
    openness = data.get("openness", 0.5)
    conscientiousness = data.get("conscientiousness", 0.5)
    extraversion = data.get("extraversion", 0.5)
    agreeableness = data.get("agreeableness", 0.5)
    neuroticism = data.get("neuroticism", 0.5)

    try:
        user_profile = np.array([openness, conscientiousness, extraversion, agreeableness, neuroticism]).reshape(1, -1)
        careers_traits = df[['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']].values

        # Calcular similitud
        similitudes = cosine_similarity(user_profile, careers_traits)[0]
        
        # Crear copia del dataframe
        df_resultado = df.copy()
        df_resultado["similitud"] = similitudes
        
        # Seleccionar las top 5 universidades más similares
        resultados = df_resultado.sort_values(by="similitud", ascending=False).head(5)
        
        # Devolver resultados
        respuesta = resultados[["id_institucion", "nombre", "tipo", "area", "escolaridad"]].to_dict(orient="records")
        
        return jsonify({
            "total_resultados": len(respuesta),
            "recomendaciones": respuesta
        }), 200
        
    except Exception as e:
        return jsonify({"Error": str(e), "message": "Verifica que las columnas de traits existan en el DataFrame"}), 500


# Ejecutar
if __name__ == "__main__":
    app.run(debug=True)
