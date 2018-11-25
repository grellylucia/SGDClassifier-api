from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
Swagger(app)
CORS(app)

@app.route('/input/task', methods=['POST'])

def predict():
    """
    Tes
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: artikel
          required:
            - teksArtikel
          properties:
            teksArtikel:
              type: string
              description: Input teks artikel.
              default:
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    teksArtikel = new_task['teksArtikel']


    X_New = teksArtikel

    clf = joblib.load('SGDClassifier.pkl')

    resultPredict = clf[0].predict(X_New)

    return jsonify({'message': format(clf[1].target_names[resultPredict])})