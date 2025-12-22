from typing import List

import numpy as np
from ES import *

# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import ProblemClass, get_problem, logger

budget = 5_000
dimension = 10
np.random.seed(42)
# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

# Hyperparameters to tune, e.g.
hyperparameter_space = {
    "mu": [1,3,5,10,20,50,100],
    "lamda": [10,20,50,100,200,500],
}

def create_problem(fid: int,parents_mu:int,LAMBDA:int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data_tune",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name=f"run_m_s_{parents_mu}_{LAMBDA}",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution strategy",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
      # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


# Hyperparameter tuning function
def tune_hyperparameters() -> List:
    # You should decide/engineer the `score` youself, which is the tuning objective
    best_score = float("inf")
    best_params = None
    # create the LABS problem and the data logger
    for pop_size in hyperparameter_space["mu"]:
        for lamda in hyperparameter_space["lamda"]:
                MU = pop_size
                LAMBDA = lamda
                F23, _logger = create_problem(23,pop_size,lamda)
                for run in range(20): 
                    studentnumber1_studentnumber2_ES(F23)
                    F23.reset() # it is necessary to reset the problem after each independent run
   
    return best_params


if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)
