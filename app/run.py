import json
import re
import plotly
import pandas as pd
import nltk


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine(f'sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine) 

# load model
model = joblib.load("models/classifier1.pkl") # Update this with the file path of model file name you want to use


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    df['is_English']=df.apply(lambda row: row.original is None, axis=1)
    df1 = genre_counts = df.groupby(['is_English','genre']).count()['message'].unstack(fill_value=0).stack()
    df2=df.iloc[:,4:-1].sum(axis = 0, skipna = True).sort_values()
    # create visuals
    graphs = [
        {
            "data": [
                        {
                        "name": "Non-English",
                        "type": "bar",
                        "x": df1.loc[0].values,
                        "y": df1.loc[0].index.values,
                        "marker": {
                                    "color": "#90EE7D"
                                    },
                        "orientation": "h"
                        },
                        {
                        "name": "English",
                        "type": "bar",
                        "x": df1.loc[1].values,
                        "y": df1.loc[1].index.values,
                        "marker": {
                                    "color": "#EE917D"
                                    },  
                        "orientation": "h"
                        }
                    ],
            "layout": {
                'title': 'Distribution of messages by genres and language',
                "barmode": "stack"
            }
        },
        {
            'data': [
                        {
                        "name": "Non-English",
                        "type": "bar",
                        "x": list(df2.index),
                        "y": list(df2.values),
                        "marker": {
                                    "color": list(df2.values)
                                    },
                        "orientation": "v"
                        }
                    ],

            'layout': {
                'title': 'Distribution of message types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()