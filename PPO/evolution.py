import torch
import numpy as np
import copy
from network import MlpPolicy
from config import EvolutionConfig

class EvolutionaryAlgorithm:
    def __init__(self, population_size=10):
        self.population_size = population_size
        self.population = []
        self.fitness_scores = []
        self.best_policy = None
        self.best_fitness = float('-inf')

    def initialize_population(self, action_size):
        """Initialize a population of random policies"""
        self.population = []
        for _ in range(self.population_size):
            policy = MlpPolicy(action_size=action_size)
            self.population.append(policy)
        self.fitness_scores = [0.0] * self.population_size

    def mutate(self, policy, mutation_rate=0.1, mutation_scale=0.1):
        """Apply Gaussian mutation to policy weights"""
        mutated_policy = copy.deepcopy(policy)
        with torch.no_grad():
            for param in mutated_policy.parameters():
                if np.random.random() < mutation_rate:
                    noise = torch.randn_like(param) * mutation_scale
                    param.add_(noise)
        return mutated_policy

    def crossover(self, parent1, parent2):
        """Perform uniform crossover between two parents"""
        child = copy.deepcopy(parent1)
        with torch.no_grad():
            for param1, param2, child_param in zip(parent1.parameters(),
                                                 parent2.parameters(),
                                                 child.parameters()):
                mask = torch.rand_like(param1) < 0.5
                child_param.copy_(torch.where(mask, param1, param2))
        return child

    def select_parents(self, tournament_size=3):
        """Tournament selection"""
        selected = []
        for _ in range(2):  # Select 2 parents
            tournament_indices = np.random.choice(len(self.population),
                                               tournament_size,
                                               replace=False)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx])
        return selected

    def evolve(self):
        """Evolve the population for one generation"""
        new_population = []

        # Elitism: Keep the best individual
        best_idx = np.argmax(self.fitness_scores)
        new_population.append(copy.deepcopy(self.population[best_idx]))

        # Generate rest of the population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Selection
            parents = self.select_parents()

            # Crossover
            child = self.crossover(parents[0], parents[1])

            # Mutation
            child = self.mutate(child)

            new_population.append(child)

        self.population = new_population
        self.fitness_scores = [0.0] * self.population_size

    def update_best_policy(self):
        """Update the best policy if a better one is found"""
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_policy = copy.deepcopy(self.population[best_idx])

    def get_best_policy(self):
        """Return the best policy found so far"""
        return self.best_policy