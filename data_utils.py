# data_utils.py

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import RMSPROP


def load_data(train_file, test_file):
    """Loads the PlantDoc dataset from the provided .npz files.
    
    Args:
        train_file: The name of the .npz file containing the training data.
        test_file: The name of the .npz file containing the testing data.

    Returns:
        A tuple containing four NumPy arrays: the training data (X_train),
        the training labels (y_train), the testing data (X_test), and
        the testing labels (y_test).
    """
    # Load the PlantDoc data
    X_train, y_train, X_test, y_test = load_data(train_file, test_file)

    # Preprocess the PlantDoc images
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Preprocess the images for use with the Xception model
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X, y):
    """Preprocesses the PlantDoc data for use with the RESNET50 model.

    This function applies the following preprocessing steps to the data:
    - Resize the images to (224, 224, 3)

    Args:
        X: A NumPy array of images, with shape (n_samples, width, height, channels).
        y: A NumPy array of labels, with shape (n_samples,).

    Returns:
        A tuple containing the preprocessed data (X) and labels (y).

    Raises:
        ValueError: if the input data is not in the expected format.
    """
    if X.ndim != 4 or y.ndim != 1:
        raise ValueError("Input data has invalid shape")

    X = np.array([np.array(Image.fromarray(img).resize((224, 224))) for img in X])
    return X, y

# Preprocess the PlantDoc images
X, y = preprocess_data(X, y)
X = preprocess_input(X)

# Load the PlantDoc data
X_train, y_train, X_test, y_test = load_data("train.npz", "test.npz")

# Create the RESNET50 model
model = ResNet50()

# Compile the model using the RMSPROP optimizer
model.compile(optimizer=RMSPROP, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model on the PlantDoc dataset
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test)
print("Test loss: {:.3f}".format(score[0]))
print("Test accuracy: {:.3f}".format(score[1]))
