from typing import List
import numpy as np

#####################################! Genetic Algorithm !#####################################
class Individual:
  def __init__(self, genomes: list) -> None:
    self.genomes = genomes
  def __len__(self) -> int:
    return len(self.genomes)
  def __eq__(self, other: 'Individual') -> bool:
    return self.genomes == other.genomes
  def __str__(self) -> str:
    return ' ==> '.join([f'[{genome}]' for genome in self.genomes])

  def fitness(self, costmat: np.ndarray) -> float:
    '''Computes fitness of individual. 

    # Parameters:
    - `costmat` - `np.ndarray`: The cost matrix.
    
    # Returns:
    - `fitness` - `float`: Fitness of individual.
    '''
    costmat_ = costmat[self.genomes][:, self.genomes]
    cost = np.sum(np.diagonal(costmat_, offset = 1))
    return 1/cost if cost != 0 else np.inf

  def mutate(self, p: float) -> None:
    '''Mutate individual. 

    # Parameters:
    - `p` - `float`: Float in [0, 1]. Mutation rate (probability).
    '''
    if np.random.random() < p:
      i, j = np.random.randint(1, len(self) - 1), np.random.randint(1, len(self) - 1)
      self.genomes[i], self.genomes[j] = self.genomes[j], self.genomes[i]
      
  @staticmethod
  def crossover(parents: List['Individual']) -> List['Individual']:
    '''Creates offspring from parents. 

    # Parameters:
    - `parents` - `List[Individual]`: An array of size 2 containing both parents.
    
    # Returns:
    - `offspring` - `List[Individual]`: An array of size 2 containing both offspring. Parents and offspring are kept the same size to avoid over-population or extinction. 
    '''
    i = np.random.randint(1, len(parents[0]) - 2)
    j = np.random.randint(i + 1, len(parents[0]) - 1)
    offspring = [[-1]*len(parents[0]), [-1]*len(parents[0])]
    for k in range(2):
      offspring[k][i:j] = parents[k].genomes[i:j]
      others = [genome for genome in parents[1 - k].genomes if genome not in offspring[k]]
      offspring[k][:i] = others[:i]
      offspring[k][j:] = others[i:]
    return [Individual(offspring[0]), Individual(offspring[1])]

class Genetic:
  def __init__(self, population_size: int, mutation_rate: float, elitism: int = None) -> None:
    '''Initializes the genetic algorithm. 

    # Parameters:
    - `population_size` - `int`: Number of individuals inside a population.
    - `mutation_rate` - `float`: A float in `[0, 1]` that indicates the mutation probability after each crossover.
    - `elitism` - `int`: Number of elite (fittest) solution to pass over to the next generation.
    
    # Returns:
    An instance of the genetic algorithm
    
    # Example:
    >>> evolution = Genetic(population_size = 100, mutation_rate = .2, elitism = 2)
    '''
    self.population_size = population_size
    self.mutation_rate = mutation_rate
    self.elitism = elitism
    self.population = None

  def initialize_population(self, genomes: list) -> List[Individual]:
    '''Samples the initial population. 

    # Parameters:
    - `genomes` - `list`: The genomes of the individuals.
    
    # Returns:
    - `population` - `List[Individual]`: List of Individuals.
    '''
    population = []
    genomes_ = np.array(genomes.copy())
    for _ in range(self.population_size):
      np.random.shuffle(genomes_[1:-1])
      population += [Individual(genomes_.copy().tolist())]
    self.population = population

  def rank_selection(self, size: int, costmat: np.ndarray, population: list = None) -> list:
    '''Select mating pool for crossover. 

    # Parameters:
    - `size` - `int`: The size of the mating pool.
    - `costmat` - `np.ndarray`: The cost matrix.
    - `population` - `list`: The population to select mating pool from.
    
    # Returns:
    - `mating_pool` - `list`: List of parents to crossover.
    '''
    population = population if population else self.population
    ranks = np.arange(1, len(population) + 1, step = 1, dtype = np.int)
    population = sorted(population, key = lambda ind: ind.fitness(costmat))
    norm_ranks = ranks/np.sum(ranks)
    mating_pool = np.random.choice(population, size = size, p = norm_ranks)
    return mating_pool.tolist()

  def simulate(self, max_generations: int, costmat: np.ndarray, verbose: bool = False):
    '''Simulates the evolution of individuals. 

    # Parameters:
    - `max_generations` - `int`: Maximum number of generations to simulate.
    - `costmat` - `np.ndarray`: The cost matrix.
    - `verbose` - `bool`: Default to `False`. Indicates whether the algorithm should report the state of generations.
    
    # Example:
    >>> num_nodes = 10
    >>> data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
    >>> costmat = distance_matrix(data, data)
    >>> evolution = Genetic(population_size = 100, mutation_rate = .2, elitism = 2)
    >>> evolution.simulate(max_generations = 30, costmat = costmat, verbose = True)
    '''
    genomes = list(range(len(costmat)))
    genomes.append(genomes[0])
    self.initialize_population(genomes)
    for gen in range(1, max_generations + 1):
      population = []
      if self.elitism:
        elites = sorted(self.population, key = lambda ind: ind.fitness(costmat))[-self.elitism:]
        population += elites
        mating_pool = self.rank_selection(len(self.population) - self.elitism, costmat, population = [ind for ind in self.population if ind not in elites])
      else:
        mating_pool = self.rank_selection(len(self.population), costmat)
      for i in range(0, len(mating_pool) - 1, 2):
        offspring = Individual.crossover(mating_pool[i:i+2])
        for child in offspring: child.mutate(self.mutation_rate)
        population += offspring
      self.population = population.copy()
      if verbose:
        print('-'*75)
        print('Generation [{:>2}/{:>2}] - Average fitness {:<6.4e} - Best fitness {:<6.4e}'.format(gen, max_generations, np.round(np.mean([ind.fitness(costmat) for ind in self.population]), 5), self.get_fittest(costmat, return_fitness = True)))

  def get_fittest(self, costmat: np.ndarray, return_fitness: bool = False):
    '''Returns the fitness/fittest individual of the current population. 

    # Parameters:
    - `costmat` - `np.ndarray`: The cost matrix.
    - `return_fitness` - `bool`: Default to `False`. If `True` returns fitness, else returns fittest individual.
    
    # Returns:
    - `individual` - `Individual`: The fittest individual of the current population.
    - `fitness` - `float`: The best fitness of the current population.
    
    # Example:
    >>> num_nodes = 10
    >>> data = np.random.randint(0, 100, size = num_nodes*2).reshape(-1, 2)
    >>> costmat = distance_matrix(data, data)
    >>> evolution = Genetic(population_size = 100, mutation_rate = .2, elitism = 2)
    >>> evolution.simulate(max_generations = 30, costmat = costmat, verbose = True)
    >>> ind = evolution.get_fittest(costmat)
    >>> path = ind.genomes
    '''
    fittest = sorted(self.population, key = lambda ind: ind.fitness(costmat))[-1]
    if return_fitness:
      return fittest.fitness(costmat)
    return fittest
  
  
#####################################! Ant Colony Algorithm !#####################################
class Ant:
  def __init__(self, nodes: np.ndarray, init_pos: int) -> None:
    self.nodes = nodes
    self.init_pos = init_pos
    self.visited = [init_pos]
    self.to_visit = [node for node in self.nodes if node != init_pos]

  def __str__(self) -> str:
    return f'Ant [Initial position = {self.init_pos}] [Nodes visited = {tuple(self.visited)}] [Nodes to visit = {(tuple(self.to_visit))}]'

  def reset(self) -> None:
    self.visited = [self.init_pos]
    self.to_visit = [node for node in self.nodes if node != self.init_pos]

  def move(self, visibility: np.ndarray, pheromones: np.ndarray, alpha: float, beta: float) -> None:
    eta = visibility[self.visited[-1], self.to_visit]
    tau = pheromones[self.visited[-1], self.to_visit]
    probs = tau**alpha * eta**beta
    probs /= np.sum(probs)
    next_node = np.random.choice(self.to_visit, p = probs)
    self.visited.append(next_node)
    self.to_visit.remove(next_node)

  def path_cost(self, costmat) -> np.float:
    costmat_ = costmat[self.visited][:, self.visited]
    diag = np.diagonal(costmat_, offset = 1)
    return np.sum(diag)

  def pheromone_intensity(self, costmat: np.ndarray, q: float) -> np.ndarray:
    pheromones = np.zeros((len(self.nodes), len(self.nodes)))
    pheromones[self.visited[:-1], self.visited[1:]] = q/self.path_cost(costmat)
    return pheromones

class AntColony:
  def __init__(self, colony_size: int, n_iter: int, alpha: float, beta: float, q: float, rho: float) -> None:
    self.colony_size = colony_size
    self.n_iter = n_iter
    self.alpha = alpha
    self.beta = beta
    self.q = q
    self.rho = rho

  def update_pheromones(self) -> None:
    self.pheromones = self.rho*self.pheromones + np.sum([ant.pheromone_intensity(self.costmat, self.q) for ant in self.ants], axis = 0)

  def fit(self, nodes: list, costmat: np.ndarray, verbose: bool = False) -> None:
    self.nodes = nodes
    self.costmat = costmat
    self.visibility = np.divide(1, costmat, where = (costmat != 0))
    self.pheromones = 1 - np.eye(len(self.nodes))
    self.ants = [Ant(self.nodes, np.random.randint(1, len(self.nodes))) for _ in range(self.colony_size)]

    for _ in range(self.n_iter):
      if verbose:
        print(f'Iteration {_ + 1}')
        print('='*120)
      for ant in self.ants:
        ant.reset()
        for __ in range(len(self.nodes) - 1):
          ant.move(self.visibility, self.pheromones, alpha = self.alpha, beta = self.beta)
        if verbose: print(ant, ant.path_cost(self.costmat))
      self.update_pheromones()
      if verbose: print()

  def get_path(self, init_pos: int = 0, cycle: bool = True, return_cost: bool = False) -> list:
    ant = sorted(self.ants, key = lambda ant: ant.path_cost(self.costmat))[0]
    path = ant.visited
    path_cost = ant.path_cost(self.costmat)
    if cycle:
      path = path + path[:1]
      path_cost += self.costmat[ant.visited[-1], ant.visited[0]]
    if return_cost:
      return path, path_cost
    return path