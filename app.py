import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pickle

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
def init():
    global classifier, sc, graph
    # load the pre-trained Keras model
    classifier = load_model('models/bankChurnPrediction1.h5')
    # load the fitted StandardScaler
    scalerfile = 'scaler.sav'
    sc = pickle.load(open(scalerfile, 'rb'))
    graph = tf.get_default_graph()

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

# Getting Parameters
def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('p1'))
    parameters.append(flask.request.args.get('p2'))
    parameters.append(flask.request.args.get('p3'))
    parameters.append(flask.request.args.get('p4'))
    parameters.append(flask.request.args.get('p5'))
    parameters.append(flask.request.args.get('p6'))
    parameters.append(flask.request.args.get('p7'))
    parameters.append(flask.request.args.get('p8'))
    parameters.append(flask.request.args.get('p9'))
    parameters.append(flask.request.args.get('p10'))
    parameters.append(flask.request.args.get('p11'))
    return parameters
    
# Cross origin support
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

# API for the prediction
@app.route("/predict", methods=["GET"])
def predict():
    # numpy horizontal vector // need to use dummy variable encodings // need feature scaling for this prediction
    with graph.as_default():
        new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
    new_prediction = (new_prediction[0] > 0.5)
    if new_prediction > 0.5:
        prediction = 'false'
    else:
        prediction = 'true'
    return sendResponse({'churnOrNot': prediction})

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(threaded=True)