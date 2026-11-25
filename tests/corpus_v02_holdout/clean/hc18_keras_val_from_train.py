# expect: none
# Looks like keras early stopping on test data, but validation is the tail of the training block.
import numpy as np
from tensorflow import keras

X = np.load("X_seq.npy")
y = np.load("y_seq.npy")
n = len(X)
tr_end, va_end = int(0.7 * n), int(0.85 * n)
X_tr, y_tr = X[:tr_end], y[:tr_end]
X_va, y_va = X[tr_end:va_end], y[tr_end:va_end]
X_te, y_te = X[va_end:], y[va_end:]

model = keras.Sequential([keras.layers.Input(shape=X.shape[1:]), keras.layers.LSTM(32), keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(X_tr, y_tr, epochs=200, batch_size=64, validation_data=(X_va, y_va), callbacks=[es])

_, acc = model.evaluate(X_te, y_te, verbose=0)
print(f"Test accuracy: {acc:.3f}")
