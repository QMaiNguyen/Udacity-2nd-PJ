import sys
import re
import numpy as np
import pandas as pd

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support

import joblib

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import sqlalchemy
from sqlalchemy import create_engine


def load_data(database_filepath):
    """Function to read table from a database and assign value to model variables.

    Args:
        database_filepath (string): location of dataset

    Returns:
        X: Independent variable array
        Y: Dependent variables array
        labels: Output column names

    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(database_filepath, engine)  
    X = df.message.values
    Y = df.iloc[:,4:].values
    labels = df.iloc[:,4:].columns
    
    return X, Y, labels


def tokenize(text):
    """Function to tokenize and clean text.

    Args:
        text (string): text that you want to tokenize

    Returns:
        clean_tokens: text tokenized and cleaned

    """
    #scan for url and replace them with "urlplaceholder"
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    #nomalize to remove special characters and marks
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    #tokenize
    tokens = word_tokenize(text) 
    #remove stop words
    tokens = [tok for tok in tokens if tok not in stopwords.words("english")] 
    #lemmatize
    lemmatizer = WordNetLemmatizer() 
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Function to build model from pipeline and GridSearchCV parameters.

    Args:
        None

    Returns:
        cv: GridSearchCV model

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)), # uncomment to try tuning this parameter
        #'tfidf__use_idf': (True, False), # uncomment to try tuning this parameter
        #'clf__estimator__max_features': ('auto', 'sqrt', 'log2'), # uncomment to try tuning this parameter
        'clf__estimator__criterion' :('gini', 'entropy'), # uncomment to try tuning this parameter
        #'clf__estimator__bootstrap' :(True, False), # uncomment to try tuning this parameter
        #'clf__estimator__class_weight' :(None,'balanced') # uncomment to try tuning this parameter
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, labels):
    """Function to evaluate model by precision, recall and fscore for each category and print them out.

    Args:
        model (object): Machine Learning fitted model
        X_test (array): Input test data
        Y_test (array): Output(s) test data
        labels (array): Output column names

    Returns:
        None

    """
    y_pred = model.predict(X_test)
    for i in range(len(Y_test[0])):
        metrics = precision_recall_fscore_support(Y_test[i], y_pred[i], average='weighted')[:3]
        label = labels[i]
        print(f"The precision, recall and f1 score of '{label}' are {metrics}")


def save_model(model, model_filepath):
    """Function to save a model into a pickle file.

    Args:
        model (object): Machine Learning fitted model
        model_filepath (string): output file path

    Returns:
        None

    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, labels = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4) # increase test_size will improve reduce training time but may reduce model accuracy
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, labels)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()