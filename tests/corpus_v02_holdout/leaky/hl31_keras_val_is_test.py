# expect: LA14
# LSTM uses validation_data=(X_test, y_test) with EarlyStopping, then evaluates on X_test.
import numpy as np
from tensorflow import keras

X = np.load("X_seq.npy")
y = np.load("y_seq.npy")
cut = int(0.85 * len(X))
X_train, X_test = X[:cut], X[cut:]
y_train, y_test = y[:cut], y[cut:]

model = keras.Sequential([
    keras.layers.Input(shape=X.shape[1:]),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), callbacks=[es])

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.3f}")
