import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tqdm import tqdm
from optimizer import Optimizer
import time
from networkFunctions import testing_predictions


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "-p":
            testing_predictions()
            exit()
        else:
            print("Given argument(s) is/are invalid")
            exit()


    print("Model creation initialized")

    # Global variables
    dataset = 'TrainingData.csv'
    bestScore = 100000.0
    # Number of times to evole the population. Original: 10
    generations = 150
    # Number of networks in each generation. Original: 20
    population = 15

    bestScoreEvolution = []

    # Parameters which will be used to create networks with random values on each key
    nn_param_choices = {
    'lstms':[1,2],
    'implementation1':[1,2],
    'units1':[2,8,16,32,64,128],
    'lstm_activation1':['tanh', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
    'recurrent_activation1':['hard_sigmoid', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
    'implementation2':[1,2],
    'units2':[2,8,16,32,64,128],
    'lstm_activation2':['tanh', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
    'recurrent_activation2':['hard_sigmoid', 'tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid'],
    'nb_layers': [1, 2, 3, 4, 5],
        'nb_neurons1': [2,8,16,32,64,128],    
        'activation1': ['tanh', 'sigmoid', 'linear', 'relu'],
        'nb_neurons2': [2,8,16,32,64,128],    
        'activation2': ['tanh', 'sigmoid', 'linear', 'relu'],
        'nb_neurons3': [2,8,16,32,64,128],    
        'activation3': ['tanh', 'sigmoid', 'linear', 'relu'],
        'nb_neurons4': [2,8,16,32,64,128],    
        'activation4': ['tanh', 'sigmoid', 'linear', 'relu'],
        'nb_neurons5': [2,8,16,32,64,128],
        'activation5': ['tanh', 'sigmoid', 'linear', 'relu'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam'],
    }
    print(f"Evolving {generations} generations with population {population}")

    ### Generate population ###

    # Initialize the optimizer algorithm and the networks that will be used
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)
    #print(networks[0].network)

    # Evolve the generation.
    for i in range(generations):
        print('\n\n','-'*80)
        print(f"Doing generation {i + 1} of {generations}")

        # Train and get accuracy for networks.
        progressBar = tqdm(total=len(networks))
        for j, network in enumerate(networks):
            print(f"\nDoing network {j+1} of population {population}")
            bestScore = network.train(dataset, bestScore)
            progressBar.update(1)
        progressBar.close()

        # Sort the networks
        networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)

        # Check best score
        print('Best score up until now:', int(1000*bestScore))
        print('-'*80)
        bestScoreEvolution.append(bestScore)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)
    networks[:1][0].print_network()
    print(bestScoreEvolution)
