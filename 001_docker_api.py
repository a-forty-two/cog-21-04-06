from flask import Flask
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.backend import set_session
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
app = Flask(__name__)
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("modelwt.h5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

@app.route('/')
def hello():
    return "<p>bye bye world</p>"

@app.route('/guess/<x>')
def predict(x):
    print(x)
    data = [ [ x ] ]
    data = np.array(data)
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        prediction = model.predict(data)
    return str(prediction)

if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5001, debug = True)
