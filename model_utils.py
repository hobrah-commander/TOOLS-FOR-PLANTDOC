# model_utils.py

import os

# Set the CUDA_VISIBLE_DEVICES environment variable to use only the first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow.compat.v1 as tf

# Check if a GPU is available
if tf.test.is_gpu_available():
    # Get the name of the GPU device
    gpu_name = tf.test.gpu_device_name()
    print(f"Using GPU: {gpu_name}")
else:
    print("No GPU available")

from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.compat.v1.keras.applications import ResNet50
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.optimizers import RMSprop
from tensorflow.compat.v1.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

# Import the necessary classes from the Keras and scikit-learn libraries
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Set the TensorFlow log level to avoid printing verbose messages
tf.logging.set_verbosity(tf.logging.ERROR)

    # Define a function that returns the learning rate for a given epoch
def lr_schedule(epoch):
    # Set the learning rate to 0.1 at epoch 0, and decrease it by a factor of 10 every 5 epochs thereafter
    return 0.1 * (0.1 ** (epoch // 5)) 

def train_model(train_file, val_file, test_file, 
                epochs=10, 
                initial_epoch=0, 
                batch_size=32, 
                lr=0.1, 
                reg=0.01):
    # Load and preprocess the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(train_file, val_file, test_file)
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Create the input tensor
    inputs = Input(shape=(224, 224, 3))

    # Create the base ResNet50 model
    base_model = ResNet50(weights="imagenet", include_top=False)(inputs)

    # Add a convolutional layer with 32 filters and a 3x3 kernel
    x = Conv2D(32, (3, 3), activation="relu")(base_model)

    # Add a max pooling layer with a 2x2 pool size
    x = MaxPooling2D((2, 2))(x)

    # Add a convolutional layer with 64 filters and a 3x3 kernel
    x = Conv2D(64, (3, 3), activation="relu")(x)

    # Add a max pooling layer with a 2x2 pool size
    x = MaxPooling2D((2, 2))(x)

    # Add a classification layer on top of the base model, with the specified regularization strength
    predictions = Dense(5, activation="softmax", kernel_regularizer=regularizers.l2(reg))(x)

    # Create the model using the functional API
    model = Model(inputs=inputs, outputs=predictions)

    # Compile the model with the RMSprop optimizer and weighted cross-entropy loss
    model.compile(
        optimizer=RMSprop(), 
        loss="categorical_crossentropy", 
        metrics=["f1_score"], 
        class_weights = {0: 1, 1: 2, 2: 10}
    )

    # Create the early stopping and learning rate reduction callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=3
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1, 
        patience=3
    )

    # Create a learning rate scheduler using the lr_schedule function
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Create a data generator for training data with random horizontal flipping and vertical shifting, using 4 threads for parallel data loading and preprocessing
    train_datagen = ImageDataGenerator(
        horizontal_flip=True, 
        vertical_shift=0.2, 
        threads=4
    )
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    # Train the model on the PlantDoc data using mini-batch learning and the Adam optimizer, with the early stopping, learning rate reduction, and learning rate scheduler callbacks
    model.fit(train_generator, epochs=epochs, initial_epoch=initial_epoch, callbacks=[early_stopping, reduce_lr, lr_scheduler])

    # Evaluate the trained model on the test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

