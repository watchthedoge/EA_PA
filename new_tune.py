import numpy as np
from ioh import get_problem, logger, ProblemClass

# ==========================
# GLOBAL SETTINGS
# ==========================
budget = 5000
dimension = 10
np.random.seed(42)

# ==========================
# RECOMBINATION
# ==========================
def recombination(parents, LAMBDA):
    MU, dimension, _ = parents.shape
    offspring = np.empty((LAMBDA, dimension, 2))

    for l in range(LAMBDA):
        idx = np.random.randint(0, MU, size=dimension)
        offspring[l, :, 0] = parents[idx, np.arange(dimension), 0]
        offspring[l, :, 1] = parents[idx, np.arange(dimension), 1]

    return offspring

# ==========================
# MUTATION
# ==========================
def mutate(offspring, tau=None, tau_prime=None):
    lam, dimension, _ = offspring.shape

    if tau is None:
        tau = 1 / np.sqrt(2 * np.sqrt(dimension))
    if tau_prime is None:
        tau_prime = 1 / np.sqrt(2 * dimension)

    global_noise = np.random.randn(lam, 1)
    local_noise = np.random.randn(lam, dimension)

    offspring[:, :, 1] *= np.exp(tau_prime * global_noise + tau * local_noise)
    offspring[:, :, 0] += offspring[:, :, 1] * np.random.randn(lam, dimension)

    return offspring

# ==========================
# (μ + λ) SELECTION
# ==========================
def selection_plus(parents, offspring, problem, MU):
    combined = np.concatenate((parents, offspring), axis=0)
    fitness = np.array([problem(combined[i, :, 0]) for i in range(len(combined))])
    idx = np.argsort(fitness)[:MU]
    return combined[idx]

# ==========================
# (μ, λ) SELECTION
# ==========================
def selection_mu_lambda(offspring, problem, MU):
    fitness = np.array([problem(offspring[i, :, 0]) for i in range(len(offspring))])
    idx = np.argsort(fitness)[:MU]
    return offspring[idx]

# ==========================
# EVOLUTION STRATEGY
# ==========================
def run_ES(problem, MU, LAMBDA, selection_type):

    if LAMBDA < MU:
        return np.nan

    parents = np.random.uniform(-5, 5, size=(MU, dimension, 2))
    parents[:, :, 1] = np.random.uniform(0.0, 1.0, size=(MU, dimension))

    while problem.state.evaluations < budget:

        offspring = recombination(parents, LAMBDA)
        offspring = mutate(offspring)

        if selection_type == "plus":
            parents = selection_plus(parents, offspring, problem, MU)
        elif selection_type == "comma":
            parents = selection_mu_lambda(offspring, problem, MU)
        else:
            raise ValueError("selection_type must be 'plus' or 'comma'")

    # ✅ RETURN IOH BEST-SO-FAR VALUE
    return problem.state.current_best.y

# ==========================
# PROBLEM CREATION
# ==========================
def create_problem(fid, selection_type, mu, lam):
    problem = get_problem(
        fid,
        dimension=dimension,
        instance=1,
        problem_class=ProblemClass.BBOB,
    )

    log = logger.Analyzer(
        root="data_bench",
        folder_name=f"run_{selection_type}_mu_{mu}_lambda_{lam}",
        algorithm_name=f"Evolution Strategy ({selection_type})",
        algorithm_info="Median of best-so-far y",
    )

    problem.attach_logger(log)
    return problem, log

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    MU_values = [1, 3, 5, 10, 20]
    LAMBDA_values = [10, 20, 50, 100, 150]
    selection_types = ["plus", "comma"]

    for MU in MU_values:
        for LAMBDA in LAMBDA_values:
            for selection_type in selection_types:

                print(f"\nRunning MU={MU}, LAMBDA={LAMBDA}, selection={selection_type}")

                problem, log = create_problem(23, selection_type, MU, LAMBDA)

                best_so_far_runs = []

                for run in range(20):
                    best_y = run_ES(problem, MU, LAMBDA, selection_type)
                    best_so_far_runs.append(best_y)
                    problem.reset()

                log.close()

                median_best_y = np.median(best_so_far_runs)

                print(
                    f"Median best-so-far y over 20 runs: "
                    f"{median_best_y:.6e}"
                )
