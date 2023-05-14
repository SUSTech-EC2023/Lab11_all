import numpy as np
import random
import matplotlib.pyplot as plt

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(weights, inputs):
    layer1_weights = weights[:6].reshape(2, 3)
    layer2_weights = weights[6:].reshape(3, 1)

    layer1 = sigmoid(np.dot(inputs, layer1_weights))
    output = sigmoid(np.dot(layer1, layer2_weights))

    return output

def fitness_function(weights):
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_outputs = np.array([[0], [1], [1], [0]])

    predictions = predict(weights, xor_inputs)
    errors = (predictions - xor_outputs) ** 2

    return -np.sum(errors) # The fitness function needs to be maximized

def generate_initial_population(population_size, chromosome_length):
    return [np.random.uniform(-1, 1, chromosome_length) for _ in range(population_size)]

def selection(population, fitness_scores, num_parents):
    scores = [float(score) for score in fitness_scores]
    sorted_population = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:num_parents]

def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        crossover_point = random.randint(1, len(parent1)-1)
        offspring.append(np.hstack((parent1[:crossover_point], parent2[crossover_point:])))

    return offspring

def mutation(offspring, mutation_rate):
    mutated_offspring = []
    for child in offspring:
        mutated_child = np.copy(child)
        for i in range(len(mutated_child)):
            if random.random() < mutation_rate:
                mutated_child[i] = np.random.uniform(-1, 1)

        mutated_offspring.append(mutated_child)

    return mutated_offspring

def genetic_algorithm(population_size, chromosome_length, num_generations, num_parents, mutation_rate):
    population = generate_initial_population(population_size, chromosome_length)
    best_fitness_scores = []

    for generation in range(num_generations):
        fitness_scores = [fitness_function(individual) for individual in population]
        best_fitness_scores.append(max(fitness_scores))
        parents = selection(population, fitness_scores, num_parents)
        offspring = crossover(parents, population_size - num_parents)
        offspring = mutation(offspring, mutation_rate)
        population = parents + offspring

    best_individual = max(population, key=fitness_function)
    return best_individual, best_fitness_scores

if __name__ == "__main__":
    population_size = 100
    chromosome_length = 9
    num_generations = 1000
    num_parents = 20
    mutation_rate = 0.1

    best_weights, best_fitness_scores = genetic_algorithm(population_size, chromosome_length, num_generations, num_parents, mutation_rate)

    # Plot the best fitness scores over generations
    plt.plot(best_fitness_scores)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score")
    plt.title("Best Fitness Score per Generation")
    plt.show()

    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = predict(best_weights, xor_inputs)

    print("Predictions:")
    print(predictions.round())
