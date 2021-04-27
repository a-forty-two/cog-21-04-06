import numpy as np
import tensorflow as tf
from tensorflow import keras

x = [ [1],[2],[3],[54],[55],[9],[10]]

y = [ [1], [0], [1], [0], [1], [1], [0] ]

x2 = np.array(x )
y2 = np.array(y )
model = keras.models.Sequential()
model.add(keras.layers.Dense(16, activation='relu',input_dim=1))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x2,y2,epochs=50)
performance = model.evaluate(x2,y2)
print(str(performance[0]))
print(str(performance[1]))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

print('predictions')
xtest = [ [42],[24] ]
predictions = model.predict(xtest)
print(predictions)
