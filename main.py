import warnings, random
warnings.filterwarnings("ignore")

import numpy as np
from sr_ga import SRwithGeneticAlgorithm
from typing import *

def f(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.exp(-2.0*x) * np.sin(20.0 * x)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    X = np.linspace(-1, 1, 1000)
    y = f(X)

    X_train, X_eval = X[:800], X[800:]
    y_train, y_eval = y[:800], y[800:]

    best_tree, best_value = SRwithGeneticAlgorithm(
        X_train, 
        y_train, 
        n_population=1000, 
        n_iter=200, 
        max_tree_depth=7, 
        complexity_weight=0.0015
    )
    print(f"Best fitness score: {best_value}\n\n")
    print(f"Analytical model: {best_tree.symbol()}\n\n")
    print(f"Best tree depth: {best_tree.depth}\n\n")
    print(f"Best tree nodes: {best_tree.n_nodes}")