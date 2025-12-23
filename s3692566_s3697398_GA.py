from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass

# To make your results reproducible (not required by the assignment), you could set the random seed by
np.random.seed(42)

# global parameters
budget = 5000
pop_size = 10
crossover_prob = 0.7
mutation_prob = 0.02


def s3692566_s3697398_GA(problem: ioh.problem.PBO) -> None:
    n = int(problem.meta_data.n_variables)
    population = np.random.randint(0, 2, size=(pop_size, n), dtype=np.int8)
    fitnesses = np.array([problem(individual) for individual in population])
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
   
    while problem.state.evaluations < budget:
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            # Crossover
            child1, child2 = one_point_crossover(parent1, parent2, crossover_prob)
            # Mutation
            child1 = mutation(child1, mutation_prob)
            child2 = mutation(child2, mutation_prob)
            new_population.append(child1)
            new_population.append(child2)
            
        # Create the new population
        population = np.array(new_population[:pop_size], dtype=np.int8) #in case of odd pop_size
        fitnesses = np.array([problem(individual) for individual in population], dtype=np.float64)
    # no return value needed
    pass

def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Tournament selection picking 3 random individuals and choosing the best one
    """
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    # find the best individual among the selected
    best_index = selected_indices[0]
    for idx in selected_indices:
        if fitnesses[idx] > fitnesses[best_index]:
            best_index = idx
    return population[best_index]

def one_point_crossover(parent1, parent2, crossover_prob):
    """
    1-point crossover at a random position, creating two children
    """
    if np.random.random() < crossover_prob:
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:])).astype(np.int8)
        child2 = np.concatenate((parent2[:point], parent1[point:])).astype(np.int8)
        return child1, child2
    else: # No crossover
        return parent1, parent2
    
def mutation(individual, mutation_prob):
    """
    Bit flip mutation
    """
    mutated = []
    for bit in individual:
        if np.random.random() < mutation_prob:
            mutated.append(1 - bit)
        else:
            mutated.append(bit)
    return np.array(mutated, dtype=np.int8)

def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        s3692566_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder
    
    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        s3692566_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()
    