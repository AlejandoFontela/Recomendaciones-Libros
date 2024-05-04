from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_redis import FlaskRedis
import redis

app = Flask(__name__)
redis_client = FlaskRedis(app)

df = pd.read_csv('BX-Books.csv', on_bad_lines='skip', encoding='latin-1', sep=';')
df.duplicated(subset='Book-Title').sum()
df = df.drop_duplicates(subset='Book-Title')
sample_size = 15000
df = df.sample(n=sample_size, replace=False, random_state=490)

def clean_text(author):
    result = str(author).lower()
    return(result.replace(' ',''))

df['Book-Author'] = df['Book-Author'].apply(clean_text)
df['Book-Title'] = df['Book-Title'].str.lower()
df['Publisher'] = df['Publisher'].str.lower()
df2 = df.drop(['ISBN','Image-URL-S','Image-URL-M','Image-URL-L','Year-Of-Publication'],axis=1)
df2['data'] = df2[df2.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df2['data'])
similarities = cosine_similarity(vectorized)
df = pd.DataFrame(similarities, columns=df['Book-Title'], index=df['Book-Title']).reset_index()

# Ruta principal para mostrar el formulario y los resultados
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_book = request.form['input_book']
        recommendations = get_recommendations(input_book)
        return render_template('index.html', recommended_books=recommendations)
    return render_template('index.html', recommended_books=None)

def get_recommendations(input_book):
    # Verificar si las recomendaciones ya están almacenadas en Redis
    stored_recommendations = redis_client.get(input_book)
    if stored_recommendations:
        return stored_recommendations.decode('utf-8').split(',')
    else:
        recommendations = pd.DataFrame(df.nlargest(11, input_book)['Book-Title'])
        recommendations = recommendations[recommendations['Book-Title'] != input_book]
        recommended_books = recommendations['Book-Title'].values.tolist()
        # Almacenar las recomendaciones en Redis para futuras consultas
        redis_client.set(input_book, ','.join(recommended_books))
        return recommended_books

if __name__ == '__main__':
    app.run(debug=True)
