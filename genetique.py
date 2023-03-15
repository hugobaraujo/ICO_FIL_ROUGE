import random
import numpy as np
import matplotlib as plt
import fil_rouge_tools as frt

clients  = frt.get_clients()

def fitness(individual):
    return frt.simulate(clients, individual)

def crossover(parent1, parent2):
    # This function performs crossover between two parents to create two children
    crossover_point = random.randint(0, len(parent1)-1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
    # This function performs mutation on an individual with the specified mutation rate
    mutated = []
    for bit in individual:
        if random.random() < mutation_rate:
            mutated.append(1 - bit)
        else:
            mutated.append(bit)
    return mutated

def select_parents(population):
    # This function selects two parents from the population using tournament selection
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=fitness), max(tournament, key=fitness)

def generate_population(population_size, genome_size):
    # This function generates a random population of the specified size and genome size
    population = []
    for i in range(population_size):
        individual =  list(np.arange(clients.shape[0])).copy()
        random.shuffle(individual)
        population.append(individual)
    return population

def genetic_algorithm(population_size, genome_size, mutation_rate, num_generations):
    # This function runs the genetic algorithm for the specified number of generations
    population = generate_population(population_size, genome_size)
    for i in range(num_generations):
        new_population = []
        for j in range(population_size // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population
    best_individual = max(population, key=fitness)
    return best_individual

# Execute:
best_individual = genetic_algorithm(200, len(clients), 0.1, 200)
print(best_individual)
print(fitness(best_individual))

frt.view_solution(clients, best_individual)
