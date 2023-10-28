import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.jamieoliver.com/recipes/category/course/mains/"
page = requests.get(url)

soup = BeautifulSoup(page.text, "html.parser")

food_cat = ["mains", "snacks", "breakfast", "desserts"]

recipe_urls = pd.Series([a.get("href") for a in soup.find_all("a")])

recipe_urls = recipe_urls[(recipe_urls.str.count("-") > 0) 
                        & (recipe_urls.str.contains("/recipes/") == True)
                        & (recipe_urls.str.contains("-recipes/") == True)
                        & (recipe_urls.str.contains("course") == False)
                        & (recipe_urls.str.contains("books") == False)
                        & (recipe_urls.str.endswith("recipes/") == False)].unique()

df = pd.DataFrame({"recipe_urls":recipe_urls})
df['recipe_urls'] = "https://www.jamieoliver.com" + df['recipe_urls'].astype('str')
recipe_url_df = df.copy()

recipe_url_df.to_csv(r"C:\Prog\Code\Estudio\UPB\5-Ingenieria-Software\FoodRecomendationSystem\src\scraping\recipe_urls.csv", sep="\t", index=False)
