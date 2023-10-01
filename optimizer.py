from network import Network
import random

class Optimizer():
    '''
    Optimizer algorithm
    '''

    def __init__(self, nn_param_choices, retain=0.4, random_select=0.1, mutate_chance=0.2):
        '''
        Initializes the Optimizer with its default parameters
        '''
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        '''
        Creates a list of "count" networks with random param choices
        '''
        networkPopulation = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_choices)
            network.create_random()
            # Add the network to our population.
            networkPopulation.append(network)
        return networkPopulation

    @staticmethod
    def fitness(network):
        return network.accuracy

    #def grade(self, population):
    #    summed = reduce(add, (self.fitness(network) for network in population))
    #    return summed / float((len(population)))

    def breed(self, mother, father):
        children = []
        for _ in range(2):
            child = {}
            # Loop through the parameters and pick params for the kid.
            for param in self.nn_param_choices:
                child[param] = random.choice([mother.network[param], father.network[param]])
            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)
            # Randomly mutate some of the children.
            if self.mutate_chance > random.random(): # 20%
                network = self.mutate(network)
            children.append(network)
        return children

    def mutate(self, network):
        # Choose a random key.
        mutation = random.choice(list(self.nn_param_choices.keys()))
        # Mutate one of the params.
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])
        return network

    def evolve(self, population):
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in population]
    
        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
    
        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain) # 40%
    
        # The parents are every network we want to keep.
        parents = graded[:retain_length]
    
        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random(): # 10%
                parents.append(individual)
    
        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []
        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:
            # Get a random mom and dad.
            #print("Parents Length: ", parents_length)
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)
    
            # Assuming they aren't the same network...
            if male != female:
                male   = parents[male]
                female = parents[female]
    
                # Breed them.
                babies = self.breed(male, female)
    
                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)
        return parents
  