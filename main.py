import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

products = pd.read_csv("productdetails.csv")
products.head(1)

products = products[['_id', 'productname', 'productcategory', 'productdescription']]

products.head(10)

print(type(products['productdescription'][0]))
products.isnull().sum()

products.duplicated().sum()

products.head(4)


def clean_text(html):
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Extract the plain text
    plain_text = soup.get_text()

    return plain_text


print(clean_text("<p>What is the difference between bluebird.js</p>"))
print(clean_text("<h1>Waqas Ahmad</h1>"))

import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


stem(
    'In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')

products['recommendation_string'] = products['productname'] + products['productdescription'] + products[
    'productcategory']


print(type(products['_id'][0]))

products['recommendation_string'] = products['recommendation_string'].apply(stem)

print(products['recommendation_string'][1])

products['recommendation_string'] = products['recommendation_string'].apply(lambda x: x.lower())


products.head()

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=10000, stop_words="english")

vectors = cv.fit_transform(products['recommendation_string']).toarray()

vectors[0]

# cv.get_feature_names()

# getting the angular distance between two vectors

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]


def recommend(product):
    product_index = products[products['productname'] == product].index[0]
    distances = similarity[product_index]
    print(distances)
    product_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]

    for i in product_list:
        print(products.iloc[i[0]].productname)


def recommend_ids(product_id):
    product_index = products[products['_id'] == product_id].index[0]
    distances = similarity[product_index]
    products_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    product_send = []

    for i in products_list:
        print(products.iloc[i[0]]._id)
        product_send.append(products.iloc[i[0]]._id)

    return product_send


ids = recommend_ids('646b848e10e18973d66f7b3c')

myrecommendation_string = products.loc[products['_id'] == '646b848e10e18973d66f7bd6', 'recommendation_string'].values[0]


from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/hello', methods=['GET'])
def responsed():
    return jsonify(['6468e3ecf13d485eda32f5e1',
                    '6468e3ecf13d485eda32f5e1'])


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        productid = request.json['productid']
        print("Message:", productid)
        response = recommend_ids(productid)  # Assuming recommend_ids() returns a list of recommended product IDs
        print(response)
        return jsonify({"recommended_products": response})
    except KeyError:
        # Handle the case when 'productid' is missing in the request JSON
        return jsonify({"error": "Missing 'productid' in the request JSON"}), 400
    except Exception as e:
        # Handle any other exceptions that may occur
        return jsonify({"error": "400 Bad Request: " + str(e)}), 400

if __name__ == '__main__':
    app.run()

