from flask import Flask, json, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter
import numpy as np
import re
import os

# Cargar modelo de spaCy para español
try:
    nlp = spacy.load("es_core_news_sm")
except:
    # Si no está instalado, usar una versión simplificada
    print("Advertencia: spaCy no está instalado. Usando lematización básica.")
    nlp = None

# Cargar datos limpios
df = pd.read_csv("Careers.csv").dropna()

# Crear app Flask
app = Flask(__name__)
CORS(app)  # Permitir CORS para el frontend

# ========== FUNCIONES DE LEMATIZACIÓN ==========

def lematizar_con_spacy(texto):
    """Lematiza texto usando spaCy (más preciso)"""
    if nlp is None:
        return lematizar_simple(texto)
    
    # Limpiar texto
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    
    # Procesar con spaCy
    doc = nlp(texto)
    
    # Extraer lemas únicos (sin stopwords y sin palabras cortas)
    lemas = []
    for token in doc:
        if not token.is_stop and len(token.text) > 2 and token.is_alpha:
            lemas.append(token.lemma_)
    
    # Contar frecuencias y devolver palabras únicas
    palabra_freq = Counter(lemas)
    return list(palabra_freq.keys())

def lematizar_simple(texto):
    """Lematización simple sin spaCy (fallback)"""
    # Palabras vacías en español
    stopwords = {
        'de', 'la', 'que', 'el', 'en', 'y', 'a', 'es', 'se', 'no', 'te', 'lo', 'le', 
        'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'un', 'las', 
        'como', 'pero', 'sus', 'una', 'está', 'han', 'ser', 'o', 'tiene', 'más',
        'muy', 'todo', 'también', 'sobre', 'esta', 'entre', 'cuando', 'hasta'
    }
    
    # Limpiar y dividir texto
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    palabras = texto.split()
    
    # Filtrar stopwords y palabras duplicadas
    palabras_unicas = []
    seen = set()
    
    for palabra in palabras:
        if palabra not in stopwords and len(palabra) > 2 and palabra not in seen:
            palabras_unicas.append(palabra)
            seen.add(palabra)
    
    return palabras_unicas

def procesar_frases_dataframe(df_subset):
    """Procesa todas las frases del dataframe y las lematiza"""
    frases_procesadas = []
    
    for idx, row in df_subset.iterrows():
        try:
            # Si la columna tiene JSON, parsearlo
            if isinstance(row.get("palabras_clave_ingreso"), str):
                frases = json.loads(row["palabras_clave_ingreso"])
            else:
                frases = row.get("palabras_clave_ingreso", [])
            
            if isinstance(frases, list):
                for frase in frases:
                    palabras_lematizadas = lematizar_con_spacy(str(frase))
                    frases_procesadas.extend(palabras_lematizadas)
        except Exception as e:
            print(f"Error procesando fila {idx}: {e}")
            continue
    
    # Contar frecuencias y devolver las más comunes
    palabra_freq = Counter(frases_procesadas)
    return palabra_freq



@app.route("/lematizar", methods=["POST"])
def lematizar():
    """Lematiza un texto enviado por el usuario"""
    data = request.get_json()
    texto = data.get("texto", "")
    
    if not texto:
        return jsonify({"error": "Texto no proporcionado"}), 400
    
    # Lematizar
    palabras_lematizadas = lematizar_con_spacy(texto)
    
    # Contar frecuencias
    palabra_freq = Counter(palabras_lematizadas)
    
    return jsonify({
        "texto_original": texto,
        "palabras_unicas": palabras_lematizadas,
        "total_palabras_unicas": len(palabras_lematizadas),
        "frecuencias": dict(palabra_freq.most_common(10))
    }), 200

@app.route("/frases", methods=["POST"])
def obtener_frases():
    """Obtiene y procesa frases de una carrera específica"""
    data = request.get_json()
    id_institucion = data.get("id_institucion", "")
    nombre_carrera = data.get("nombre_carrera", "")
    
    if not id_institucion or not nombre_carrera:
        return jsonify({"error": "id_institucion y nombre_carrera son requeridos"}), 400
    
    # Filtrar dataframe
    df_filtrado = df[(df["id_institucion"] == id_institucion) & 
                     (df["nombre"] == nombre_carrera)]
    
    if df_filtrado.empty:
        return jsonify({"error": "No se encontró la carrera especificada"}), 404
    
    # Procesar frases
    palabra_freq = procesar_frases_dataframe(df_filtrado)
    
    # Obtener las 20 palabras más comunes
    palabras_top = palabra_freq.most_common(20)
    
    return jsonify({
        "id_institucion": id_institucion,
        "nombre_carrera": nombre_carrera,
        "palabras_clave": [palabra for palabra, _ in palabras_top],
        "frecuencias": dict(palabras_top),
        "total_palabras_unicas": len(palabra_freq)
    }), 200

@app.route("/generar-test", methods=["POST"])
def generar_test():
    """Genera pares de palabras para el test vocacional"""
    data = request.get_json()
    carreras = data.get("carreras", [])  # Lista de nombres de carreras
    num_pares = data.get("num_pares", 15)  # Número de pares a generar
    
    if not carreras:
        return jsonify({"error": "Lista de carreras requerida"}), 400
    
    # Obtener palabras clave de cada carrera
    palabras_por_carrera = {}
    
    for carrera in carreras:
        df_carrera = df[df["nombre"] == carrera]
        
        if not df_carrera.empty:
            palabra_freq = procesar_frases_dataframe(df_carrera)
            # Obtener top 10 palabras de esta carrera
            palabras_por_carrera[carrera] = [palabra for palabra, _ in palabra_freq.most_common(10)]
    
    if len(palabras_por_carrera) < 2:
        return jsonify({"error": "Se necesitan al menos 2 carreras con palabras clave"}), 400
    
    # Generar pares de palabras
    import random
    pares = []
    carreras_list = list(palabras_por_carrera.keys())
    
    for i in range(num_pares):
        # Seleccionar dos carreras diferentes
        carrera1, carrera2 = random.sample(carreras_list, 2)
        
        # Seleccionar una palabra de cada carrera
        if palabras_por_carrera[carrera1] and palabras_por_carrera[carrera2]:
            palabra1 = random.choice(palabras_por_carrera[carrera1])
            palabra2 = random.choice(palabras_por_carrera[carrera2])
            
            pares.append({
                "id": i + 1,
                "palabra1": palabra1,
                "palabra2": palabra2,
                "categoria1": carrera1,
                "categoria2": carrera2
            })
    
    return jsonify({
        "total_pares": len(pares),
        "pares": pares
    }), 200



# ========== RECOMENDACIÓN BIG FIVE ==========
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




# ========== EJECUTAR APP ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)