import unidecode
import ast
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from ing_cleaning import ingredient_parser
import config
from flask import Flask, request, jsonify

def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

def get_recommendations(N, scores):
    df_recipes = pd.read_csv(config.PARSED_PATH)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    recommendation = pd.DataFrame(columns=['recipe', 'ingredients', 'score', 'url'])
    count = 0
    for i in top:
        recommendation.at[count, 'recipe'] = title_parser(df_recipes.recipe_name[i])
        recommendation.at[count, 'ingredients'] = ingredient_parser_final(df_recipes.ingredients[i])
        recommendation.at[count, 'recipe_urls'] = df_recipes.recipe_urls[i]
        recommendation.at[count, 'score'] = f"{scores[i]}"
        count += 1
    return recommendation

def title_parser(title):
    title = unidecode.unidecode(title)
    return title

def ingredient_parser_final(ingredient):
    if isinstance(ingredient, list): ingredients = ingredient
    else: ingredients = ast.literal_eval(ingredient)
    
    ingredients = ",".join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

class MeanEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.vector_size = word_model.wv.vector_size
    
    def fit(self): return self
    
    def transform(self, docs):
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector
    
    def word_average(self, sent):
        mean = []
        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(self.word_model.wv.get_vector(word))
        if not mean: return np.zeros(self.vector_size)
        else: return np.array(mean).mean(axis=0)
    
    def word_average_list(self, docs): return np.vstack([self.word_average(sent) for sent in docs])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size
    
    def fit(self, docs):
        text_docs = []
        for doc in docs: text_docs.append(" ".join(doc))
        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)
        max_idf = max(tfidf.idf_)
        self.word_idf_weight = defaultdict(lambda: max_idf, [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
        return self
    
    def transform(self, docs):
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector
    
    def word_average(self, sent):
        mean = []
        for word in sent:
            if word in self.word_model.wv.index_to_key:
                mean.append(self.word_model.wv.get_vector(word) * self.word_idf_weight[word])
        if not mean: return np.zeros(self.vector_size)
        else: return np.array(mean).mean(axis=0)
    
    def word_average_list(self, docs): return np.vstack([self.word_average(sent) for sent in docs])

def get_rec(ingredients, N = 5, mean = False):
    model = Word2Vec.load(config.MODEL_PATH)
    if model: print('Model loaded successfully')
    data = pd.read_csv(config.PARSED_PATH)
    data['parsed'] = data.ingredients.apply(ingredient_parser)
    corpus = get_and_sort_corpus(data)
    
    if mean:
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)
    else:
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)
    
    inp = ingredients
    inp = inp.split(',')
    inp = ingredient_parser(inp)
    
    if mean: input_embedding = mean_vec_tr.transform([inp])[0].reshape(1, -1)
    else: input_embedding = tfidf_vec_tr.transform([inp])[0].reshape(1, -1)
    
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    
    recommendations = get_recommendations(N, scores)
    return recommendations[['recipe', 'ingredients', 'score', 'recipe_urls']]

api = Flask(__name__)

@api.route('/request', methods=['GET'])
def send_rec():
    ingredients = request.headers.get('ingredients')
    print(ingredients)
    #drop url and score
    rec = get_rec(ingredients)
    print(rec.to_dict(orient='records'))
    return jsonify(rec.to_dict(orient='records'))

if __name__ == '__main__':
    api.run(debug=True, host='0.0.0.0', port=6969)