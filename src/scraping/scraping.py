import pandas as pd
import time
import numpy as np
from jo_scraping import JamieOliver

recipe_df = pd.read_csv('./recipe_urls.csv')
attribs = ['recipe_name', 'serves', 'cooking_time', 'difficulty', 'ingredients']

temp = pd.DataFrame(columns=attribs)
for i in range(0, len(recipe_df['recipe_urls'])):
    url = recipe_df['recipe_urls'][i]
    recipe_scraper = JamieOliver(url)
    temp.loc[i] = [getattr(recipe_scraper, attrib)() for attrib in attribs]
    if i % 50 == 0:
        print(f'Step {i} completed')
    time.sleep(np.random.randint(1, 5))

temp['recipe_urls'] = recipe_df['recipe_urls']
columns = ['recipe_urls'] + attribs
temp = temp[columns]
temp.to_csv(r"C:\Prog\Code\Estudio\UPB\5-Ingenieria-Software\FoodRecomendationSystem\src\scraping\recipes.csv", index=False)