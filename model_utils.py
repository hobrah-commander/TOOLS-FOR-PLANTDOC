# model_utils.py

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from data_utils import load_data, preprocess_data

def train_model(train_file, test_file, epochs=10):
    # Load and preprocess the data
    X_train, y_train, X_test, y_test = load_data(train_file, test_file)
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Create the base Xception model
    base_model = Xception(weights="imagenet", include_top=False)

    # Add a classification layer on top of the base model
    x = base_model.output
    x = Dense(128, activation="relu")(x)
    predictions = Dense(5, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model on the PlantDoc data
    model.fit(X_train, y_train, epochs=epochs)

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)
