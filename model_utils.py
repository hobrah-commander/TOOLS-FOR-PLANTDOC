#model_utils.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

def train_model(train_file, test_file, epochs=10, initial_epoch=0, batch_size=32):
    # Import the EarlyStopping and ReduceLROnPlateau callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Load and preprocess the data
    X_train, y_train, X_test, y_test = load_data(train_file, test_file)
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Create the base Xception model
    base_model = Xception(weights="imagenet", include_top=False)

    # Add a classification layer on top of the base model
    x = base_model.output
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5, training=True)(x)
    predictions = Dense(5, activation="softmax", kernel_regularizer=l2(0.01))(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])

    # Create the early stopping and learning rate reduction callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    # Create a data generator for training data with random horizontal flipping and vertical shifting
    train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_shift=0.2)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    # Train the model on the PlantDoc data using mini-batch learning and stochastic gradient descent, with the early stopping and learning rate reduction callbacks
    model.fit(train_generator, epochs=epochs, initial_epoch=initial_epoch, shuffle=True, callbacks=[early_stopping, reduce_lr])

    # Add dropout with a dropout rate of 0.5 and training=False (during evaluation and prediction)
    model.add(Dropout(0.5, training=False))

    # Evaluate the model on the test data
    test_loss, test_acc = model.fit(X_test, y_test)
    print("Test accuracy:", test_acc)
