import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, make_scorer

import pickle

def load_data(database_filepath):
    '''
    X: message strings
    Y: message categories, binary representation, multi categories possible.
    category_names: the names of categories, a list of strings.
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Message',engine)
    X = df.message
    Y = df.drop(['index', 'id','message','original','genre'],axis = 1)
    category_names = list(Y.columns)
    return (X, Y, category_names)


def tokenize(text):
    # convert text to lower case and remove punctuation
    text = re.sub('[^0-9a-zA-Z]', ' ', text.lower())
    
    # tokenize words
    tokens = word_tokenize(text)
    
    # remove stopwords and lemmatize
    stemmer = PorterStemmer()
    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words('english'):
            clean_tokens.append(stemmer.stem(tok))
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC()))
    ])
    parameters = {
    'vect__ngram_range':[(1,1),(1,2)],
    'tfidf__norm': ['l1','l2'],
    'clf__estimator__loss':['squared_hinge','hinge']
    }
    # using the average f1 score for all categories for gridsearch score
    scorer = make_scorer(average_f1)
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, verbose = 3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints the precision recall f1-score and support of each categories.
    '''
    Y_pred = model.predict(X_test)
    metrics = []
    for i in range(len(category_names)):
        report = classification_report(np.array(Y_test)[:, i], Y_pred[:, i])
        metrics.append(report)
        print(category_names[i], '\n', metrics[i])
        
def average_f1(Y_test, Y_pred):
    '''
    returns the average f1 scores of all categories, 
    categories where no positive is predicted will have a 0 score.
    '''
    metrics = []
    for i in range(Y_test.shape[1]):
        if np.sum(Y_pred[:,i]) == 0:
            report = 0
        else:
            report = f1_score(np.array(Y_test)[:, i], Y_pred[:, i])
        metrics.append(report)
    return np.mean(metrics)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
            
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)





    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()