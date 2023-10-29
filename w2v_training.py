from gensim.models import Word2Vec
import pandas as pd
import sys
sys.path.insert(1, './src')
import config
import ast

def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.ingredients_parsed.values:
        doc = doc.split(' ')
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

def get_window(corpus):
    lengths = [len(doc) for doc in corpus]
    avg_length = float(sum(lengths) / len(lengths))
    return round(avg_length)

if __name__ == '__main__':
    data = pd.read_csv('./src/clean_recipes.csv')
    corpus = get_and_sort_corpus(data)
    print(f'Corpus length: {len(corpus)}')
    model_cbow = Word2Vec(corpus, min_count=1, vector_size=100, window=get_window(corpus), workers=8, sg=0)
    model_cbow.save('./model_cbow.bin')