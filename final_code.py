from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

# we receive the train and test data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
# normalization of the pixels(0 to 1 range)
X_train = X_train/255
X_test = X_test/255
# changing it to to a list form from the matrix form
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
# building the model
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                  input_shape=(32, 32, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), padding='same',
                  activation='relu'),
    layers.MaxPooling2D(),

    # dense network layer(for classification)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='sigmoid')
])

# using tensorboard for plotting (loss vs epochs) and (accuracy vs epochs)
tb_callback=tf.keras.callbacks.TensorBoard(log_dir = 'logs/', histogram_freq = 1)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, callbacks=[tb_callback])
model.summary()
print('this is the accuracy of the test data: ')
model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
yp = [np.argmax(i) for i in y_pred]

# different graphs and summary
print(classification_report(y_test, yp))
cm = confusion_matrix(y_test, yp)
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('predicted')
plt.ylabel('true')
plt.title('confusion matrix')
plt.show()


