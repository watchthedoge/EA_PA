import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

MU = 5
LAMBDA =20

budget = 5000
dimension = 10

np.random.seed(42)

#HELPER FUNCTIONS
def recombination(parents, LAMBDA):
    """
    parents: shape (MU, dimension, 2)
             parents[:, :, 0] = x
             parents[:, :, 1] = sigma
    LAMBDA: number of offspring

    returns: offspring of shape (LAMBDA, dimension, 2)
    """
    MU, dimension, n_components = parents.shape
    assert n_components == 2

    offspring = np.empty((LAMBDA, dimension, 2))

    for l in range(LAMBDA):
        # choose one parent per dimension
        parent_idx = np.random.randint(0, MU, size=dimension)

        # copy (x_i, sigma_i) together
        offspring[l, :, 0] = parents[parent_idx, np.arange(dimension), 0]
        offspring[l, :, 1] = parents[parent_idx, np.arange(dimension), 1]

    return offspring
def mutate(offspring, tau=None, tau_prime=None, sigma_min=1e-8):
    """
    offspring: shape (LAMBDA, dimension, 2)
               offspring[:, :, 0] = x
               offspring[:, :, 1] = sigma

    mutates offspring IN PLACE and returns it
    """
    lam, dimension, _ = offspring.shape

    # default learning rates (standard ES choice)
    if tau is None:
        tau = 1 / np.sqrt(2 * np.sqrt(dimension))
    if tau_prime is None:
        tau_prime = 1 / np.sqrt(2 * dimension)

    # mutate sigma global is for each offspring and local is for each x of such offspring 
    global_sample = np.random.standard_normal((lam, 1))
    local_sample = np.random.standard_normal((lam, dimension))

    offspring[:, :, 1] *= np.exp(
        tau_prime * global_sample + tau * local_sample
    )

    # optional prevent sigma from becoming too small
    # offspring[:, :, 1] = np.maximum(offspring[:, :, 1], sigma_min)

    # mutate x 
    offspring[:, :, 0] += offspring[:, :, 1] * np.random.standard_normal((lam, dimension))

    return offspring

#Greedy selection (mu + lambda)
def selection_plus(parents, offspring,problem):
    """
    parents: shape (MU, dimension, 2)
             parents[:, :, 0] = x
             parents[:, :, 1] = sigma
    offspring: shape (LAMBDA, dimension, 2)
               offspring[:, :, 0] = x
               offspring[:, :, 1] = sigma
    fitness_parents: shape (MU,)
    """
    # Combine parents and offspring
    combined_parents_offspring = np.concatenate((parents, offspring), axis=0)
    
    # Compute fitness for combined population
    combined_fitness = np.array([problem(combined_parents_offspring[i, :, 0]) for i in range(len(combined_parents_offspring))])
    
    # Select the best individuals from the combined population
    selected_indices = np.argsort(combined_fitness)[:MU]
    
    # Return the selected parents and their fitness values
    return combined_parents_offspring[selected_indices], combined_fitness[selected_indices]


def selection_mu_lambda_elitist(parents, fitness_parents, offspring, MU, problem):
    """
    Weakly elitist (μ,λ)-ES selection
    """
    # Evaluate offspring
    fitness_offspring = np.array([
        problem(offspring[i, :, 0]) for i in range(len(offspring))
    ])

    # Select best MU offspring
    selected_indices = np.argsort(fitness_offspring)[:MU]
    new_parents = offspring[selected_indices]
    new_fitness = fitness_offspring[selected_indices]

    #  elitism: keep best parent if better 
    best_parent_idx = np.argmin(fitness_parents)
    worst_new_idx = np.argmax(new_fitness)

    if fitness_parents[best_parent_idx] < new_fitness[worst_new_idx]:
        new_parents[worst_new_idx] = parents[best_parent_idx]
        new_fitness[worst_new_idx] = fitness_parents[best_parent_idx]

    return new_parents, new_fitness



#Greedy selection (mu, lambda)
def selection_mu_lambda(offspring, MU, problem):
    fitness = []
    evaluated_indices = []

    for i in range(len(offspring)):
        if problem.state.evaluations >= budget:
            break
        fitness.append(problem(offspring[i, :, 0]))
        evaluated_indices.append(i)

    fitness = np.array(fitness)
    evaluated_indices = np.array(evaluated_indices)

    # select best MU among evaluated offspring
    selected = np.argsort(fitness)[:min(MU, len(fitness))]
    return offspring[evaluated_indices[selected]], fitness[selected]






########################################################################################
def studentnumber1_studentnumber2_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    
    #SIGMA = 0.5
    parents = np.random.uniform(low=-5, high=5, size=(MU, dimension, 2))#setting initial parents using the problem bounds
    parents[:, :, -1] = np.random.uniform(0.0, 1.0, size=(MU, dimension))  # setting the initial sigma to random values between 0 and 1
    counter = 0
    while problem.state.evaluations < budget:   
        
        #recombination- offspring generation
        offspring = recombination(parents, LAMBDA)
        #mutation
        mutated_offspring = mutate(offspring)
        #ensure mutated offspring are within bounds

        #selection
        parents, fitness_array_parents = selection_mu_lambda(mutated_offspring, MU, problem)
        counter += 1
    # no return value needed 


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution strategy",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F23, _logger = create_problem(23)
    for run in range(20): 
        studentnumber1_studentnumber2_ES(F23)
        F23.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder


