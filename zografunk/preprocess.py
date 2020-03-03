import pandas as pd
import re
import glob
from tqdm import tqdm


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

    df.to_csv(SAVE_PATH, index=False, sep=':')
