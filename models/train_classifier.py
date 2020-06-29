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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
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
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    metrics = []
    for i in range(len(category_names)):
        report = classification_report(np.array(Y_test)[:, i], Y_pred[:, i])
        metrics.append(report)
        print(category_names[i], '\n', metrics[i])


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
        #with open(model_filepath, 'rb') as file:
            #model = pickle.load(file)
            
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