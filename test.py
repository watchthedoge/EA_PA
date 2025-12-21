import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

budget = 50000
dimension = 10



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

problem, _logger = create_problem(23)
MU = 10
LAMBDA = 50
SIGMA = 0.5
parents = np.random.uniform(low=-5, high=5, size=(MU, dimension, 2))#setting initial parents using the problem bounds
parents[:, :, -1] = np.random.uniform(0.0, 1.0, size=(MU, dimension))  # setting the initial sigma to random values between 0 and 1

fitness = np.array([
    problem(parents[i, :, 0])
    for i in range(MU)
])



#recombiantion-offspring generation)
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


import numpy as np

def mutate(offspring, tau=None, tau_prime=None, sigma_min=1e-8):
    """
    offspring: shape (LAMBDA, dimension, 2)
               offspring[:, :, 0] = x
               offspring[:, :, 1] = sigma

    mutates offspring IN PLACE and returns it
    """
    LAMBDA, dimension, _ = offspring.shape

    # default learning rates (standard ES choice)
    if tau is None:
        tau = 1 / np.sqrt(2 * np.sqrt(dimension))
    if tau_prime is None:
        tau_prime = 1 / np.sqrt(2 * dimension)

    # mutate sigma global is for each offspring and local is for each x of such offspring 
    global_sample = np.random.standard_normal((LAMBDA, 1))
    local_sample = np.random.standard_normal((LAMBDA, dimension))

    offspring[:, :, 1] *= np.exp(
        tau_prime * global_sample + tau * local_sample
    )

    # optional prevent sigma from becoming too small
    # offspring[:, :, 1] = np.maximum(offspring[:, :, 1], sigma_min)

    # mutate x 
    offspring[:, :, 0] += offspring[:, :, 1] * np.random.standard_normal((LAMBDA, dimension))

    return offspring

print("mutation test:")
print("before mutation:")

offspring = recombination(parents, LAMBDA)
print(offspring[1])
mutated_offspring = mutate(offspring)

print("after mutation:")
print(mutated_offspring[1])



#Greedy selection
def selection_mu_lambda(offspring, MU):
    """
    offspring: shape (LAMBDA, dimension, 2)
    returns: new parents of shape (MU, dimension, 2)
    """
    fitness = np.array([
        problem(offspring[i, :, 0]) for i in range(len(offspring))
    ])

    selected_indices = np.argsort(fitness)[:MU]
    return offspring[selected_indices], fitness[selected_indices]
print("selection test:")
selected_parents, selected_fitness = selection_mu_lambda(mutated_offspring, MU)
print("selected parents' fitness:")
print(selected_fitness)
print("selected parents:")
print(selected_parents)