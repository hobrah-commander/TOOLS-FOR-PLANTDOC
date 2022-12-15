# visualizations_utils.py

import matplotlib.pyplot as plt

# Define a function that visualizes the learning rate and regularization parameters for evolutionary training
def visualize_evolutionary_training(history):
    # Extract the learning rate and regularization parameters from the history object
    lrs = [x['lr'] for x in history]
    regs = [x['reg'] for x in history]

    # Create a scatter plot of the learning rate and regularization parameters
    plt.scatter(lrs, regs)
    plt.xlabel('Learning Rate')
    plt.ylabel('Regularization')
    plt.title('Evolutionary Training')
    plt.show()

# Define a function that visualizes the performance of the evolutionary algorithm
def visualize_evolutionary_performance(history):
    # Extract the training and validation accuracy from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Create a scatter plot of the training and validation accuracy
    plt.scatter(acc, val_acc)
    plt.xlabel('Training Accuracy')
    plt.ylabel('Validation Accuracy')
    plt.title('Evolutionary Performance')
    plt.show()
    
# Define a function that visualizes the training and validation accuracy and loss
def visualize_training(history):
    # Extract the training and validation accuracy and loss from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create subplots for the training and validation accuracy and loss
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(acc)
    axs[0, 0].plot(val_acc)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_title('Training and Validation Accuracy')
    axs[0, 0].legend(['Train', 'Val'])
    axs[0, 1].plot(loss)
    axs[0, 1].plot(val_loss)
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Training and Validation Loss')
    axs[0, 1].legend(['Train', 'Val'])

    # Show the plots
    plt.show()
    
# Define a function that visualizes the training and validation precision
def visualize_precision(history):
    # Extract the training and validation precision from the history object
    precision = history.history['precision']
    val_precision = history.history['val_precision']

    # Create a line plot of the training and validation precision over the course of training
    plt.plot(precision)
    plt.plot(val_precision)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend(['Train', 'Val'])
    plt.show()

# Define a function that visualizes the training and validation accuracy
def visualize_accuracy(history):
    # Extract the training and validation accuracy from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Create a line plot of the training and validation accuracy over the course of training
    plt.plot(acc)
    plt.plot(val_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(['Train', 'Val'])
    plt.show()

# Define a function that visualizes the training and validation loss
def visualize_loss(history):
    # Extract the training and validation loss from the history object
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create a line plot of the training and validation loss over the course of training
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(['Train', 'Val'])
    plt.show()

# Define a function that visualizes the model's performance on the test set
def visualize_test_performance(model, X_test, y_test):
    # Evaluate the model on the test set
    metrics = model.evaluate(X_test, y_test, verbose=0)

    # Create a bar plot of the model's performance on the test set
    plt.bar(['Accuracy', 'Precision'], [metrics[1], metrics[2]])
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Test Performance')
    plt.show()
