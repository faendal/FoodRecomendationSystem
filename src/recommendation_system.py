import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ing_cleaning import ingredient_parser
import pickle
import config
import unidecode
import ast

def get_recommendations(N, scores):
    df_recipes = pd.read_csv(config.PARSED_PATH)
    top = sorted(range(len(scores)), key = lambda i: scores[i], reverse=True)[:N]
    recommendation = pd.DataFrame(columns= ['recipe', 'ingredients', 'score', 'url'])
    count = 0
    for i in top:
        recommendation.at[count, 'recipe'] = title_parser(df_recipes['recipe_name'][i])
        recommendation.at[count, 'ingredients'] = ingredient_parser_final(df_recipes['ingredients'][i])
        recommendation.at[count, 'url'] = df_recipes['recipe_urls'][i]
        recommendation.at[count, 'score'] = "{:.3f}".format(float(scores[i]))
        count += 1
    return recommendation

def ingredient_parser_final(ingredient):
    if isinstance(ingredient, list): ingredients = ingredient
    else: ingredients = ast.literal_eval(ingredient)
    
    ingredients = ','.join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def title_parser(title):
    title = unidecode.unidecode(title)
    return title

def rec_sys(ingredients, N = 5):
    
    with open(config.TFIDF_ENCODING_PATH, 'rb') as f: tfidf_encodings = pickle.load(f)
    with open(config.TFIDF_MODEL_PATH, 'rb') as f: tfidf = pickle.load(f)
    
    try: ingredients_parsed = ingredient_parser(ingredients)
    except: ingredients_parsed = ingredient_parser([ingredients])
    
    ingredients_tfidf = tfidf.transform([ingredients_parsed])
    
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)
    
    recommendations = get_recommendations(N, scores)
    return recommendations