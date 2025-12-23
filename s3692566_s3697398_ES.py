import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

# GLOBAL SETTINGS
MU = 5
LAMBDA = 20

budget = 5000
dimension = 10

#random seed 42 for reproducibility
np.random.seed(42)

#HELPER FUNCTIONS
def recombination(parents, LAMBDA):
    """
    Implements uniform global discrete recombination

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

def mutate(offspring, tau=None, tau_prime=None):
    """
    offspring: shape (LAMBDA, dimension, 2)
               offspring[:, :, 0] = x
               offspring[:, :, 1] = sigma

    mutates offspring IN PLACE and returns it
    """
    lam, dimension, _ = offspring.shape

    # default learning rates if not provided, implementated following lecture notes
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

    # mutate x 
    offspring[:, :, 0] += offspring[:, :, 1] * np.random.standard_normal((lam, dimension))

    return offspring
'''
#Elitist selection (mu + lambda)
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
    return combined_parents_offspring[selected_indices]
'''


#selection (mu, lambda)
def selection_mu_lambda(offspring, MU, problem):
    fitness = []
    evaluated_indices = []

    for i in range(len(offspring)):
        # break if budget is exceeded
        if problem.state.evaluations >= budget:
            break
        fitness.append(problem(offspring[i, :, 0]))
        evaluated_indices.append(i)

    fitness = np.array(fitness)
    evaluated_indices = np.array(evaluated_indices)

    # select best MU among evaluated offspring
    selected = np.argsort(fitness)[:min(MU, len(fitness))]
    return offspring[evaluated_indices[selected]]


########################################################################################

########################################################################################

# EVOLUTION STRATEGY IMPLEMENTATION

def s3692566_s3697398_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    

    parents = np.random.uniform(low=-5, high=5, size=(MU, dimension, 2))#setting initial parents using the problem bounds
    parents[:, :, -1] = np.random.uniform(0.0, 1.0, size=(MU, dimension))  # setting the initial sigma to random values between 0 and 1
    counter = 0
    while problem.state.evaluations < budget:   
        
        #recombination- offspring generation
        offspring = recombination(parents, LAMBDA)
        #mutation
        mutated_offspring = mutate(offspring)
        #selection + fitness evaluation (in selection function)
        parents = selection_mu_lambda(mutated_offspring, MU, problem)
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
        s3692566_s3697398_ES(F23)
        F23.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder


