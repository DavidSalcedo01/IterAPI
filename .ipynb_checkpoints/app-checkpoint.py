from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
df = pd.read_csv("data.csv")  # debe contener columnas como: id_institucion, nombre, estado, carreras
df["contenido"] = df["estado"].fillna("") + " " + df["nombre_y"].fillna("")

# Vectorizar texto
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["contenido"])

# Crear app Flask
app = Flask(__name__)

@app.route("/recomendar", methods=["POST"])
def recomendar():
    data = request.get_json()
    interes = data.get("interes", "")

    if not interes:
        return jsonify({"error": "Interés no proporcionado"}), 400

    # Vectorizar el interés del usuario
    interes_vectorizado = vectorizer.transform([interes])

    # Calcular similitud
    similitudes = cosine_similarity(interes_vectorizado, tfidf_matrix)[0]
    df["similitud"] = similitudes

    # Seleccionar las top 5 universidades más similares
    resultados = df.sort_values(by="similitud", ascending=False).head(5)

    # Devolver resultados como JSON
    respuesta = resultados[["nombre_x", "estado", "nombre_y", "similitud"]].to_dict(orient="records")
    return jsonify(respuesta)

# Ejecutar
if __name__ == "__main__":
    app.run(debug=True)
