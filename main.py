import pandas as pd
import tensorflow as tf
from time import strftime, localtime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow.keras.backend as K


K.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def preprocessing():
    df = pd.read_csv("heart.csv")

    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    df = df[(df["ca"] < 4) & (df["thal"] > 0)]
    targets = df["target"]
    features = df.drop(columns=["target"])
    features = pd.get_dummies(features, columns=cat_cols)

    # standardize numerical features
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2, random_state=1)
    scaler = StandardScaler().fit(features_train[num_cols])
    features_train.loc[:, features_train.columns.isin(num_cols)] = scaler.transform(features_train[num_cols])
    features_test.loc[:, features_test.columns.isin(num_cols)] = scaler.transform(features_test[num_cols])

    return features_train, features_test, targets_train, targets_test


def build_model(input_dim):
    model = Sequential([
        Dense(128, activation="relu", kernel_initializer="he_uniform", input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.1),
        Dense(256, activation="relu", kernel_initializer="he_uniform"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation="sigmoid", kernel_initializer="he_uniform")
    ])

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
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
        epochs=100,
        batch_size=32,
        verbose=2,
        shuffle=True,
        validation_data=(x_test, y_test)
        # callbacks=[
        #     tensorboard,
        #     checkpoint
        # ]
    )


if __name__ == "__main__":
    train()
