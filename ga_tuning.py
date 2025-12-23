import argparse
import csv
from typing import Tuple
import numpy as np
import ioh
from ioh import get_problem, ProblemClass
import GA

np.random.seed(42)
total_evals = 0

def evaluate_params(pop_size, crossover_prob, mutation_prob):
    global total_evals
    runs = 3
    scores_18 = []
    scores_23 = []
    GA.pop_size = pop_size
    GA.crossover_prob = crossover_prob
    GA.mutation_prob = mutation_prob
    GA.budget = 1300
    for r in range(runs):
        problem18 = get_problem(18, dimension=50, instance=1, problem_class=ProblemClass.PBO)
        GA.s3692566_studentnumber2_GA(problem18)
        scores_18.append(float(problem18.state.current_best.y))
        total_evals += problem18.state.evaluations

        problem23 = get_problem(23, dimension=49, instance=1, problem_class=ProblemClass.PBO)
        GA.s3692566_studentnumber2_GA(problem23)
        scores_23.append(float(problem23.state.current_best.y))
        total_evals += problem23.state.evaluations

    scores_arr_18 = np.array(scores_18, dtype=float)
    scores_arr_23 = np.array(scores_23, dtype=float)
    return float(scores_arr_18.mean()), float(scores_arr_23.mean())


def grid_search():
    pop_sizes = [10, 20, 40]
    crossover_probs = [0.5, 0.6, 0.7, 0.8]
    mutation_prob = 0.02

    results = []
    for p in pop_sizes:
        for c in crossover_probs:
            mean_score_18, mean_score_23 = evaluate_params(pop_size=p, crossover_prob=c, mutation_prob=mutation_prob)
            #print(f'Config: pop_size={p}, crossover_prob={c} => Mean Best 18: {mean_score_18:.4f}, Mean Best 23: {mean_score_23:.4f}')
            results.append({
                "pop_size": p,
                "crossover_prob": c,
                "mean_best_F18": mean_score_18,
                "mean_best_F23": mean_score_23,
            })         
    return results

def rank_based_selection(results):
    s18 = np.array([r["mean_best_F18"] for r in results], dtype=float)
    s23 = np.array([r["mean_best_F23"] for r in results], dtype=float)
    ordered18 = np.argsort(s18)[::-1]  # returns indexes when sorting in descending order (largest to smallest value)
    ordered23 = np.argsort(s23)[::-1]
    rank18 = np.zeros(len(ordered18), dtype=int)
    rank23 = np.zeros(len(ordered23), dtype=int)
    # ranks of the fitness scores
    rank18[ordered18] = np.arange(1, len(results) + 1) 
    rank23[ordered23] = np.arange(1, len(results) + 1)
    
    # average ranks between both problems
    avg_rank = (rank18 + rank23) / 2.0
    best_idx = np.argmin(avg_rank)
    best_params = results[best_idx]
    
    return best_params["pop_size"], best_params["crossover_prob"]
    
if __name__ == "__main__":
    results = grid_search()
    pop_size, crossover_prob = rank_based_selection(results)
    print(f'Best parameters found: pop_size={pop_size}, crossover_prob={crossover_prob}')
    print(f'Total evaluations across all runs: {total_evals}')