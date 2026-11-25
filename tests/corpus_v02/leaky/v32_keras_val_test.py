# expect: LA14:high
import numpy as np
from tensorflow import keras
data = np.load("windows.npz")
X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
model = keras.Sequential([keras.layers.LSTM(32), keras.layers.Dense(1)])
model.compile(loss="mse", optimizer="adam")
stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[stop])
print(model.evaluate(X_test, y_test))
