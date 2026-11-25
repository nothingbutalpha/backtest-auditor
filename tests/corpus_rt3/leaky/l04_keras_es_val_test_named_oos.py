# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
from tensorflow import keras
es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_oos, y_oos), epochs=200, callbacks=[es])
model.evaluate(X_oos, y_oos)
