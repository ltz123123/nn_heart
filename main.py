import pandas as pd
import tensorflow as tf
from time import strftime, localtime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


def preprocessing():
    df = pd.read_csv("heart.csv")

    targets = df["target"]
    features = df.drop(columns=["ca", "thal", "target"])

    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    # create dummies for categorical features
    features = pd.get_dummies(features, columns=cat_cols)

    # standardize numerical columns
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2)
    scaler = StandardScaler().fit(features_train[num_cols])
    features_train[num_cols] = scaler.transform(features_train[num_cols])
    features_test[num_cols] = scaler.transform(features_test[num_cols])

    return features_train, features_test, targets_train, targets_test


def build_model(input_dim):
    model = Sequential([
        Dense(256, activation="relu", input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(lr=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train():
    x_train, x_test, y_train, y_test = preprocessing()
    model = build_model(x_train.shape[1])

    time_now = strftime("%Y_%m_%d_%H_%M", localtime())
    NAME = f"heart_disease_model_{time_now}"
    tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))
    checkpoint = ModelCheckpoint(
        "saved_models/{}.model".format("heart_data_{epoch:02d}_{val_loss:.3f}"),
        monitor="val_loss",
        mode="min"
    )

    model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        verbose=2,
        shuffle=True,
        callbacks=[
            tensorboard,
            # checkpoint
        ]
    )

    model.evaluate(
        x_test, y_test,
        verbose=2
    )


if __name__ == "__main__":
    train()
