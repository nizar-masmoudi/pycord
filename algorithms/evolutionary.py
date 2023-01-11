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

  def fitness(self, cost_matrix: np.ndarray) -> np.float:
    cost_matrix_ = cost_matrix[self.genomes][:, self.genomes]
    cost = np.sum(np.diagonal(cost_matrix_, offset = 1))
    return 1/cost if cost != 0 else np.inf

  def mutate(self, p: float) -> None:
    if np.random.random() < p:
      i, j = np.random.randint(1, len(self) - 1), np.random.randint(1, len(self) - 1)
      self.genomes[i], self.genomes[j] = self.genomes[j], self.genomes[i]
      
  @staticmethod
  def crossover(parents: List['Individual']) -> List['Individual']:
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
    self.population_size = population_size
    self.mutation_rate = mutation_rate
    self.elitism = elitism
    self.population = None

  def initialize_population(self, genomes) -> List[Individual]:
    population = []
    genomes_ = np.array(genomes.copy())
    for _ in range(self.population_size):
      np.random.shuffle(genomes_[1:-1])
      population += [Individual(genomes_.copy().tolist())]
    self.population = population

  def rank_selection(self, size: int, cost_matrix: np.ndarray, population: list = None) -> list:
    population = population if population else self.population
    ranks = np.arange(1, len(population) + 1, step = 1, dtype = np.int)
    population = sorted(population, key = lambda ind: ind.fitness(cost_matrix))
    norm_ranks = ranks/np.sum(ranks)
    mating_pool = np.random.choice(population, size = size, p = norm_ranks)
    return mating_pool.tolist()

  def fit(self, genomes: np.ndarray, max_generations: int, cost_matrix: np.ndarray, verbose: bool = False):
    self.initialize_population(genomes)
    if verbose:
      print()
      print('|{:^10}   {:^10}   {:^10}'.format('Generation', 'Mean cost', 'Least cost'))
      print('-'*38)
    for _ in range(max_generations):
      population = []
      if self.elitism:
        elites = sorted(self.population, key = lambda ind: ind.fitness(cost_matrix))[-self.elitism:]
        population += elites
        mating_pool = self.rank_selection(len(self.population) - self.elitism, cost_matrix, population = [ind for ind in self.population if ind not in elites])
      else:
        mating_pool = self.rank_selection(len(self.population), cost_matrix)
      for i in range(0, len(mating_pool) - 1, 2):
        offspring = Individual.crossover(mating_pool[i:i+2])
        for child in offspring: child.mutate(self.mutation_rate)
        population += offspring
      self.population = population.copy()
      if verbose:
        print('|{:^10}   {:^10.2f}   {:^10.2f}|'.format(_ + 1, np.round(np.mean([1/ind.fitness(cost_matrix) for ind in self.population]), 5), 1/self.get_fittest(cost_matrix, return_fitness=True)))
        print('-'*38)

  def get_fittest(self, cost_matrix: np.ndarray, return_fitness: bool = False):
    fittest = sorted(self.population, key = lambda ind: ind.fitness(cost_matrix))[-1]
    if return_fitness:
      return fittest.fitness(cost_matrix)
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

  def path_cost(self, cost_matrix) -> np.float:
    cost_matrix_ = cost_matrix[self.visited][:, self.visited]
    diag = np.diagonal(cost_matrix_, offset = 1)
    return np.sum(diag)

  def pheromone_intensity(self, cost_matrix: np.ndarray, q: float) -> np.ndarray:
    pheromones = np.zeros((len(self.nodes), len(self.nodes)))
    pheromones[self.visited[:-1], self.visited[1:]] = q/self.path_cost(cost_matrix)
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
    self.pheromones = self.rho*self.pheromones + np.sum([ant.pheromone_intensity(self.cost_matrix, self.q) for ant in self.ants], axis = 0)

  def fit(self, nodes: list, cost_matrix: np.ndarray, verbose: bool = False) -> None:
    self.nodes = nodes
    self.cost_matrix = cost_matrix
    self.visibility = np.divide(1, cost_matrix, where = (cost_matrix != 0))
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
        if verbose: print(ant, ant.path_cost(self.cost_matrix))
      self.update_pheromones()
      if verbose: print()

  def get_path(self, init_pos: int = 0, cycle: bool = True, return_cost: bool = False) -> list:
    ant = sorted(self.ants, key = lambda ant: ant.path_cost(self.cost_matrix))[0]
    path = ant.visited
    path_cost = ant.path_cost(self.cost_matrix)
    if cycle:
      path = path + path[:1]
      path_cost += self.cost_matrix[ant.visited[-1], ant.visited[0]]
    if return_cost:
      return path, path_cost
    return path