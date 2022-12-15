# model_utils.py

import os
import time
import random

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

# Import Deap for Evolutionary Training    
import deap
from deap import base, creator, tools

    # Generate a random seed based on the current time
    random_seed = int(time.time())

    # Use the random seed to initialize the random number generator
    random.seed(random_seed)

    # Set the random seed to ensure reproducibility
    RANDOM_SEED = random_seed
    np.random.seed(RANDOM_SEED)

#Import visualization_utils
import visualizations_utils

# Define a function to generate random hyperparameters
def generate_hyperparameters():
    return (toolbox.lr(), toolbox.reg())

    # Define an individual as a list of hyperparameters
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define the population as a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define the mutation and crossover operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.05)

    # Define the selection operator
    toolbox.register("select", tools.selTournament, tournsize=3)

# Define the evolutionary training function
def evolutionary_train(model, X_train, y_train, X_val, y_val, epochs, population_size=10, n_generations=10):
    # Initialize the population
    population = toolbox.population(n=population_size)

    # Evolve the population
    for generation in range(n_generations):
        # Apply the evolutionary operators to generate a new population
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)

        # Evaluate the fitness of the individuals in the offspring
        fitnesses = model.evaluate(offspring, X_train, y_train, X_val, y_val, epochs=epochs)

        # Select the best individuals from the current and next generations
        population = toolbox.select(population + offspring, k=population_size)

    # Return the best individual in the final generation
    return population

# Define a function to perform grid search
def grid_search(X_train, y_train, X_val, y_val, epochs, population_size=10, n_generations=10):
    # Create a KerasClassifier object
    model = KerasClassifier(build_fn=create_model, epochs=epochs, verbose=0)

    # Define the grid search parameters
    hyperparameters = generate_hyperparameters()
    param_grid = dict(lr=hyperparameters[0], reg=hyperparameters[1])

    # Perform the grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, y_train)

    # Print the grid search results
            print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print(f"mean={mean:.4f}, std={stdev:.4f} with: {param}")


# Define a function that returns the learning rate for a given epoch
def lr_schedule(epoch):
    # Set the learning rate to 0.1 at epoch 0, and decrease it by a factor of 10 every 5 epochs thereafter
    return 0.1 * (0.1 ** (epoch // 5)) 

def train_model(train_file, val_file, test_file, 
                epochs=10, 
                initial_epoch=0, 
                batch_size=32, 
                lr=0.1, 
                reg=0.01,
                evolutionary=False):
    
    # Load and preprocess the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(train_file, val_file, test_file)
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    X_test, y_test = preprocess_data(X_test, y_test)
    
    # Define the model
    model = create_model(lr=lr, reg=reg)

    # Define the learning rate scheduler
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)

    # Define the callbacks
    callbacks = [lr_scheduler]

# Define a function to generate random hyperparameters
def generate_hyperparameters():
    return (toolbox.lr(), toolbox.reg())

    # Define an individual as a list of hyperparameters
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define the population as a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define the mutation and crossover operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.05)

    # Define the selection operator
    toolbox.register("select", tools.selTournament, tournsize=3)
    
 # If evolutionary training is enabled, use Deap to evolve the hyperparameters
    if evolutionary:
        
        # Define the evolutionary algorithm
        toolbox = base.Toolbox()

        # Define the search space
        toolbox.register("lr", uniform, 0.01, 0.1)
        toolbox.register("reg", uniform, 0.001, 0.1)
        
        # Define an individual as a list of hyperparameters
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Define the population as a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the fitness function
def evaluate_model(individual):
            lr, reg = individual
            model = create_model(lr=lr, reg=reg)

# Define the main evolutionary loop
def main():
    # Generate the initial population
    population = toolbox.population(n=100)

    # Evaluate the fitness of each individual
    fitnesses = [toolbox.evaluate(ind) for ind in population]
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Perform the evolutionary loop
    for g in range(100):
        # Select the next generation of individuals
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:
                toolbox.mate(child1, child2)
            toolbox.mutate(child1)
            toolbox.mutate(child2)
            del child1.fitness.values
            del child2.fitness.values

        # Evaluate the fitness of each offspring
        fitnesses = [toolbox.evaluate(ind) for ind in offspring]
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Replace the current population with the offspring
        population = offspring

    # Return the best individual
    return tools.selBest(population, k=1)[0]

if __name__ == "__main__":
    best_hyperparameters = main()
    
    # Define a fitness function that evaluates the model on the validation set
    def fitness(hyperparameters):
        # Unpack the hyperparameters
        lr, reg = hyperparameters

    # Create the input tensor
    inputs = Input(shape=(224, 224, 3))
    
    # Define a function to create the base model
    def create_base_model(input_tensor):
    base_model = ResNet50(weights="imagenet", include_top=False)(input_tensor)
    return base_model
    
    def add_convolutional_layers(base_model):
        
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
    
    # Fit the model with the learning rate scheduler, using the same callbacks as before
    history = model.fit(
    X_train, y_train,
    epochs=epochs,
    initial_epoch=initial_epoch,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr, lr_scheduler],
    validation_data=(X_val, y_val),
    workers=4
    )
    
    # Create a data generator for training data with random horizontal flipping and vertical shifting, using 4 threads for parallel data generation
    train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True
    )
    
    train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size,
    shuffle=True,
    workers=4
    )
    
   # Fit the model using the data generator, using the same callbacks as before
    history = model.fit_generator(
    train_generator,
    initial_epoch=initial_epoch,
    epochs=epochs,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // batch_size,
    )
    
    # Train the model on the PlantDoc data using mini-batch learning and the Adam optimizer, with the early stopping, learning rate reduction, and learning rate scheduler callbacks
    model.fit(train_generator, epochs=epochs, initial_epoch=initial_epoch, callbacks=[early_stopping, reduce_lr, lr_scheduler])

    # Evaluate the trained model on the test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
