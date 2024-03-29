import numpy as np
from node import Node
from tree import ExpressionBinaryTree
from typing import *

def create_forest(n_forests: 100, max_depth: int=5) -> np.ndarray:
    return np.asarray([ExpressionBinaryTree(max_depth) for i in range(0, n_forests)])

def evaluate_forest(
        forest: np.ndarray,
        X: np.ndarray, 
        y: np.ndarray, 
        complexity_weight: float=0.1) -> np.ndarray:
    
    evaluate = lambda tree : tree.evaluate(X, y, complexity_weight=complexity_weight)
    scores = np.vectorize(evaluate)(forest)
    return scores

def select_trees(
        forest: np.ndarray, 
        fitness_scores: np.ndarray, 
        n_best: int) -> np.ndarray:
    
    elite_idx = fitness_scores.argsort()[:n_best]
    return forest[elite_idx]

def crossover(forest: np.ndarray, Pc: float=0.9) -> np.ndarray:
    np.random.shuffle(forest)
    parents1 = forest[slice(0, None, 2)]
    parents2 = forest[slice(1, None, 2)]

    children1, children2 = [], []
    if len(parents1) > len(parents2):
        parents1 = parents1[:-1]
    
    dud_tree = ExpressionBinaryTree(1, Node(operator_type=0))
    cross = lambda p1, p2 : p1.cross_with(p2) if Pc > np.random.random() else (dud_tree, dud_tree)
    children1, children2 = np.vectorize(cross)(parents1, parents2)
    children = np.concatenate([children1, children2], axis=0)
    children = children[children != dud_tree]
    return children

def mutate(forest: np.ndarray, Pm: float=0.1) -> np.ndarray:
    mut = lambda tree : tree.mutate_tree(Pm=0.2) if Pm > np.random.random() else tree
    forest = np.vectorize(mut)(forest)
    return forest

def SRwithGeneticAlgorithm(
        X: np.ndarray, 
        y: np.ndarray,
        n_population: int, 
        n_iter: int=100, 
        max_tree_depth: int=7,
        Pm: float=0.1, 
        Pc: float=0.9,
        complexity_weight: float=0.001,
        verbosity: int=2
    ) -> Tuple[ExpressionBinaryTree, float]:
    
    assert n_population >= 10

    forest = create_forest(n_population, max_depth=max_tree_depth)

    best_value, best_tree = np.inf, None

    logging = lambda i, score : print(f"Generation {i+1} => Fitness score: {score}")
    for i in range(0, n_iter):
        fitness_scores = evaluate_forest(forest, X, y, complexity_weight=complexity_weight)
        parent_trees = select_trees(forest, fitness_scores, n_population//2)
        elite_tree = parent_trees[0:1][0]
        elite_evaluation = elite_tree.fitness

        if elite_evaluation < best_value:
            best_value = elite_evaluation
            best_tree = elite_tree
        
        if verbosity == 0: 
            pass

        elif verbosity == 1:
            if elite_evaluation < best_value:
                logging(i, elite_evaluation)
                
        elif verbosity == 2: logging(i, elite_evaluation)

        children_trees = crossover(parent_trees, Pc=Pc)
        children_trees = mutate(children_trees, Pm=Pm)
        forest = np.concatenate([parent_trees, children_trees], axis=0)

    return best_tree, best_value