import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_CSV = "mitbih_train.csv"
TEST_CSV  = "mitbih_test.csv"


train_df = pd.read_csv(TRAIN_CSV, header=None)
test_df  = pd.read_csv(TEST_CSV, header=None)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.astype(int)

X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:, -1].values.astype(int)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]


counts = pd.Series(y_train).value_counts().sort_index()
print("\nClass distribution (train):")
for i, c in counts.items():
    print(f"{i} ({class_names[i]}): {c}")

plt.figure()
plt.bar([class_names[i] for i in counts.index], counts.values)
plt.xticks(rotation=20, ha="right")
plt.title("Class Distribution in Training Set")
plt.tight_layout()
plt.show()

rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)

param_grid = {
    "n_estimators": [100],          
    "max_depth": [None],            
    "min_samples_split": [2, 5],    
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1_weighted",  
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

print("\nBest RF params:", grid.best_params_)

rf_pred = best_rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest test accuracy: {rf_acc*100:.2f}%")

print("\nRandom Forest classification report:")
print(classification_report(y_test, rf_pred, target_names=class_names, digits=2))

rf_cm = confusion_matrix(y_test, rf_pred)
print("\nRandom Forest confusion matrix:\n", rf_cm)

plt.figure()
plt.imshow(rf_cm)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(range(len(class_names)), class_names, rotation=20, ha="right")
plt.yticks(range(len(class_names)), class_names)
plt.colorbar()
plt.tight_layout()
plt.show()

smote = SMOTE(random_state=SEED)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("X_train_sm:", X_train_sm.shape, "y_train_sm:", y_train_sm.shape)
print(pd.Series(y_train_sm).value_counts().sort_index())

X_train_sm_cnn = X_train_sm[..., np.newaxis].astype(np.float32) 
X_test_cnn     = X_test[..., np.newaxis].astype(np.float32)


num_classes = 5
y_train_sm_oh = tf.keras.utils.to_categorical(y_train_sm, num_classes=num_classes)
y_test_oh     = tf.keras.utils.to_categorical(y_test,     num_classes=num_classes)

def build_cnn(input_shape=(187, 1), num_classes=5):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

cnn = build_cnn()
cnn.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
]

history = cnn.fit(
    X_train_sm_cnn, y_train_sm_oh,
    validation_split=0.2,
    epochs=30,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_acc = cnn.evaluate(X_test_cnn, y_test_oh, verbose=0)
print(f"\nCNN test accuracy: {test_acc*100:.2f}%")

# Predictions -> classification report
cnn_prob = cnn.predict(X_test_cnn, verbose=0)
cnn_pred = np.argmax(cnn_prob, axis=1)

print("\nCNN classification report:")
print(classification_report(y_test, cnn_pred, target_names=class_names, digits=2))

cnn_cm = confusion_matrix(y_test, cnn_pred)
print("\nCNN confusion matrix:\n", cnn_cm)

plt.figure()
plt.imshow(cnn_cm)
plt.title("Confusion Matrix - CNN (with SMOTE)")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(range(len(class_names)), class_names, rotation=20, ha="right")
plt.yticks(range(len(class_names)), class_names)
plt.colorbar()
plt.tight_layout()
plt.show()
