import pandas as pd
import re
import glob
from tqdm import tqdm

import nltk
from nltk.stem import WordNetLemmatizer


# Configure
DATA_PATH = '../data/'
files = [file for file in glob.glob(f'{DATA_PATH}*.json')]

for file in tqdm(files):

    # Read as DataFrame
    df = pd.read_json(file, dtype={'id': 'int64', 'ingredients': 'str'})

    # regex to match
    reg = re.compile('(\[|\]|\')')

    # Remove brackets, commas and quotes
    df.ingredients.replace(reg, '', inplace=True)

    # xx.csv
    SAVE_PATH = DATA_PATH + file.split('\\')[-1].split('.')[0] + '.csv'

    # df.to_csv(SAVE_PATH, index=False, sep=':')


# OR

df = pd.read_json('../data/train.json',  dtype={'id': 'int64', 'cuisine': 'str', 'ingredients': 'str'})
df = df[df.ingredients.str.len() > 1]

df['ingredients'] = df.ingredients.str.lower()

# Regex to match
# reg = re.compile('(\[|\]|\'\d\%)')
reg = re.compile('[^\w\s\,]')
df.ingredients.replace(reg, '', inplace=True)

lemmatizer = WordNetLemmatizer()
train_df['seperated_ingredients'] = train_df.seperated_ingredients.apply(lambda x: ''.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]))
print(train_df.shape)
