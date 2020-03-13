import nltk
from nltk.stem import WordNetLemmatizer

import pandas as pd
from joblib import dump

# Scikit Imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import re

print('\nReading and cleaning Dataset..')

df = pd.read_json('../data/train.json',  dtype={'id': 'int64', 'cuisine': 'str', 'ingredients': 'str'})
df = df[df.ingredients.str.len() > 1]

df['ingredients'] = df.ingredients.str.lower()

# Regex to match
# reg = re.compile('(\[|\]|\'\d\%)')
reg = re.compile('[^\w\s\,]')
df.ingredients.replace(reg, '', inplace=True)

lemmatizer = WordNetLemmatizer()
df['ingredients'] = df.ingredients.apply(lambda x: lemmatizer.lemmatize(x))

# Split Dataset into test and train, using 80-20 ratio.
X_train, X_test, y_train, y_test = train_test_split(df.ingredients, df.cuisine, test_size=0.2, random_state=42)


def tokenizer(text):
    return nltk.tokenize.casual.TweetTokenizer().tokenize(text)


# Construct a pipeline in order to use vectorizer => transformer => classifier easier.
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenizer, max_features=None,
                             encoding='utf-8', lowercase=False)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', SGDClassifier(loss='hinge', random_state=42, penalty='elasticnet')),
])

# Tuning parameters.
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__min_df': (1, 5, 10),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.0001, 0.00001),
    'clf__penalty': ('l1', 'l2', 'elasticnet'),
    # 'clf__loss': ('log', 'modified_huber', 'epsilon_insensitive', 'perceptron')
}


if __name__ == '__main__':
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)

    print(f'Training ...')

    grid_search.fit(X_train, y_train)
    print(f'\nK-fold Score: {grid_search.best_score_:.2f}.')

    best_parameters = grid_search.best_estimator_.get_params()
    print('Best Parameters:\n')
    for param_name in sorted(parameters.keys()):
        print(f'{param_name}: {best_parameters[param_name]}')

    # dump(grid_search.best_estimator_, '../trained_pipeline.pkl', compress=0)

    acc = (grid_search.predict(X_test) == y_test).mean()
    print(f'Accuracy: {acc*100:0.2f} %.')
