import copy, sympy
import numpy as np
from collections import deque
from typing import *
from operators import *
from node import Node


class ExpressionBinaryTree:
    def __init__(self, max_depth: int, root: Optional[Node]=None):
        self.max_depth = max_depth
        self.depth = 1
        self.n_nodes = 1

        if root is None:
            self.root = Node(operator_type=np.random.randint(1, 3), level=1)
            self._random_expand_tree()
        else:
            self.root = root
            Node.reset_levels(self.root, level=1)
            self.depth = self._get_tree_depth()
            self.n_nodes = len(self.to_list())

    def _get_tree_depth(self) -> int:
        open_set = deque()
        closed_set = set()
        open_set.appendleft(self.root)
        depth = 0

        while len(open_set) != 0:
            node: Node = open_set.pop()
            
            if node in closed_set: continue

            if node.left is not None:
                open_set.appendleft(node.left)

            if node.right is not None: 
                open_set.appendleft(node.right)
            
            depth = node.level
            closed_set.add(node)
        return depth

    def _transverse(self, node: Node, __symbols_list: List=[]) -> List[Union[str, int, float]]:
        if node is None:
            return __symbols_list
        
        __symbols_list.append(node.value)
        self._transverse(node.left, __symbols_list)
        self._transverse(node.right, __symbols_list)
        return __symbols_list
    
    def to_list(self) -> List[Union[str, int, float]]:
        return self._transverse(self.root, [])
    
    def _random_expand_tree(self):
        if self.root.left is not None or self.root.right is not None:
            raise Exception("Only childless trees can be expanded")

        open_set = deque()
        closed_set = set()
        open_set.appendleft(self.root)
        depth = self.root.level
        selected_max_depth = np.random.randint(2,  self.max_depth+1)

        while (len(open_set) != 0):
            node: Node = open_set.pop()
            if node in closed_set: continue
            # to avoid logarithm of zero or negative number
            if node.parent is not None and node.parent.value == "log":
                if isinstance(node.value, (int, float)):
                    node.value = abs(node.value) + (
                        np.random.uniform(1e-10, 0.1) 
                        if node.value <= 0 else node.value
                    )
            # to avoid division by zero (0)
            if node.parent is not None and node.parent.value == "/":
                if node.parent.right == node and node.value == 0:
                    node.value = node.value + np.random.uniform(1e-10, 0.1)
                    
            if node.left is None and node.right is None:
                op_start, op_stop = (1, 3) if depth < selected_max_depth-1 else (0, 1)
                if node.operator_type == 0: pass
                if node.operator_type == 1:
                    node.set_child(
                        Node(
                            operator_type=np.random.randint(op_start, op_stop)
                        ), left_or_right="left")
                    self.n_nodes += 1
                if node.operator_type == 2:
                    node.set_child(
                        Node(
                            operator_type=np.random.randint(op_start, op_stop)
                        ), left_or_right="left")
                    
                    node.set_child(
                        Node(
                            operator_type=np.random.randint(op_start, op_stop)
                        ), left_or_right="right")
                    self.n_nodes += 2

                if depth < selected_max_depth:
                    open_set.appendleft(node.left)
                    if node.right is not None:
                        open_set.appendleft(node.right)
                    depth = node.left.level
                else: continue
            self.depth = depth
            closed_set.add(node)

    def _mutate_node(self, node: Node, Pm: float=0.1):
        if node is None: return
        if Pm > np.random.random():
            if node.operator_type == 0:
                if isinstance(node.value, (int, float)):
                    node.value += np.random.normal(0, 1)
                else:
                    randint = np.random.randint(0, 2)
                    if randint == 0: 
                        node.value = np.random.choice(NULLARY_VARIABLES)
                    else:
                        node.value = np.random.uniform(*NULLARY_PARAMS_INTERVAL)
            elif node.operator_type == 1:
                node.value = np.random.choice(UNARY_OPERATORS)
            elif node.operator_type == 2:
                node.value = np.random.choice(BINARY_OPERATORS)
            else: raise ValueError(f"{node.operator_type} is an invalid operator type")

        self._mutate_node(node.left, Pm)
        self._mutate_node(node.right, Pm)
        return
    
    def mutate_tree(self, Pm: float=0.1) -> "ExpressionBinaryTree":
        self._mutate_node(self.root, Pm=Pm)
        return self

    def _random_cut_from_tree(self, tree: "ExpressionBinaryTree", depth: int) -> Node:
        child_node = tree.root
        i = 1
        while i < depth:
            if child_node.left is None and child_node.right is None: 
                break

            child = np.random.choice(["left", "right"])
            if getattr(child_node, child) is None:
                if child == "left": child = "right"
                if child == "right": child = "left"

            if getattr(child_node, child).operator_type == 0: break
            child_node = getattr(child_node, child)
            i += 1
        return child_node
    
    def _prune(self, node: Node, depth: int=1):
        if depth <= self.max_depth and node.operator_type == 0: return
        if depth == self.max_depth:                    
            if node.operator_type == 1:
                node.replace_with_child("left")
            elif node.operator_type == 2:
                node.replace_with_child(np.random.choice(["left", "right"]))
            self._prune(node, depth)
            depth -= 1
        if node.left is not None: self._prune(node.left, depth=depth+1)
        if node.right is not None: self._prune(node.right, depth=depth+1)
        return

    @classmethod
    def create_tree(cls, node: Node, max_depth: int) -> "ExpressionBinaryTree":
        return cls(root=node, max_depth=max_depth)

    def cross_with(self, other_tree: "ExpressionBinaryTree") -> Tuple["ExpressionBinaryTree", "ExpressionBinaryTree"]:
        parent1 = copy.deepcopy(self)
        parent2 = copy.deepcopy(other_tree)

        if parent1.depth == 1 or parent2.depth == 1:
            return parent1, parent2
        
        node1 = self._random_cut_from_tree(parent1, np.random.randint(1, parent1.depth))
        node2 = self._random_cut_from_tree(parent2, np.random.randint(1, parent2.depth))
        parent1 = node1.parent
        parent2 = node2.parent

        # cross parent 1 and parent 2
        node1.parent = parent2
        if node1.parent is not None:
            if node1.parent.left == node2: node1.parent.left = node1
            elif node1.parent.right == node2: node1.parent.right = node1

        # cross parent 2 and parent 1
        node2.parent = parent1
        if node2.parent is not None:
            if node2.parent.left == node1: node2.parent.left = node2
            elif node2.parent.right == node1: node2.parent.right = node2

        child1_node = node1.get_progenitor()
        child2_node = node2.get_progenitor()
        Node.reset_levels(child1_node, level=1)
        Node.reset_levels(child2_node, level=1)
        self._prune(child1_node, depth=1)
        self._prune(child2_node, depth=1)

        child1 = self.create_tree(child1_node, self.max_depth)
        child2 = self.create_tree(child2_node, self.max_depth)
        return child1, child2

    def _estimate_with_tree(self, node: Node, X: np.ndarray) -> np.ndarray:
        if node.operator_type == 0:
            if node.value in NULLARY_VARIABLES:
                return X.reshape(X.shape[0], -1)[:, NULLARY_VARIABLES.index(node.value)]
            else:
                return node.value

        left_eval = self._estimate_with_tree(node.left, X)
        if node.right is not None:
            right_eval = self._estimate_with_tree(node.right, X)

        if node.operator_type == 1:
            return getattr(np, node.value)(left_eval)
        
        if node.operator_type == 2:
            try:
                return BINARY_OPERATORS_FUNC_DICT[node.value](left_eval, right_eval)
            except (OverflowError, ZeroDivisionError):
                return np.ones(X.shape[0]) * np.inf
        
    def estimate(self, X: np.ndarray) -> np.ndarray:
        estimates = self._estimate_with_tree(self.root, X)
        if isinstance(estimates, (int, float, complex)):
            estimates = np.ones((X.shape[0], ), dtype=X.dtype) * estimates
        return estimates
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, complexity_weight: float=0.0) -> float:
        y_est = self.estimate(X)
        if np.linalg.norm(y_est[1:] - y_est[:-1]) == 0:
            return np.inf
        squared_error = (y_est - y)**2
        squared_error[squared_error != squared_error] = np.inf
        squared_error[squared_error < 0] = np.inf
        rmse = np.sqrt(squared_error.mean())
        complexity = self.n_nodes
        self.fitness = (complexity_weight * complexity) + rmse
        return self.fitness
    
    def _to_symbol(self, node: Node, symbol: str=""):
        if node is None: return str()
        if node.operator_type == 0:
            symbol += str(node.value)
            return symbol

        left_symbol = self._to_symbol(node.left, symbol) 
        if node.operator_type == 1:
            node_symbol = f"{node.value}({left_symbol})"
            return node_symbol

        if node.operator_type == 2:
            right_symbol = self._to_symbol(node.right, symbol)

            if node.value == "pow": 
                node_symbol = f"({left_symbol}**{right_symbol})"
            else:
                node_symbol = f"({left_symbol} {node.value} {right_symbol})"
            return node_symbol
        
    def symbol(self) -> Any:
        symbol = self._to_symbol(self.root, "")
        return sympy.sympify(symbol)