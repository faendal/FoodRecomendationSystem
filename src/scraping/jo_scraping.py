import requests
from bs4 import BeautifulSoup
import numpy as np

class JamieOliver:
    
    def __init__(self, url: str) -> None:
        self.url = url
        self.headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"}
        self.soup = BeautifulSoup(requests.get(self.url, headers = self.headers).content, 'html.parser')
    
    def recipe_name(self):
        try:
            return self.soup.find('h1').text.strip()
        except:
            return np.nan
    
    def serves(self):
        try:
            return self.soup.find('div', {'class' : 'recipe-detail serves'}).text.split(' ', 1)[1].strip()
        except:
            return np.nan
    
    def cooking_time(self):
        try:
            return self.soup.find('div', {'class' : 'recipe-detail time'}).text.split('In')[1].strip()
        except:
            return np.nan
    
    def difficulty(self):
        try:
            return self.soup.find('div', {'class' : 'col-md-12 recipe-details-col remove-left-col-padding-md'}).text.split('Difficulty')[1].strip()
        except:
            return np.nan
    
    def ingredients(self):
        try:
            ingredients = []
            for li in self.soup.select('.ingred-list li'):
                ingred = ' '.join(li.text.split())
                ingredients.append(ingred)
            return ingredients
        except:
            return np.nan

if __name__ == "__main__":
    url = 'https://www.jamieoliver.com/recipes/vegetables-recipes/veggie-chilli/'
    recipe_scraper = JamieOliver(url)
    print(recipe_scraper.recipe_name())
    print(recipe_scraper.serves())
    print(recipe_scraper.cooking_time())
    print(recipe_scraper.difficulty())
    print(recipe_scraper.ingredients())