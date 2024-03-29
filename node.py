import numpy as np
from typing import *
from operators import *

class Node:
    def __init__(
            self, 
            value: Optional[Union[str, float]]=None, 
            operator_type: Optional[str]=None, 
            level: Optional[int]=None):
        
        c1 = ((value is None) and (operator_type is None))
        c2 = ((value is not None) and (operator_type is not None))
        if c1 or c2:
            raise ValueError(
                "Either value or operator_type should be defined, but not both at the same time"
            )

        self.value = value
        self.level = level
        self.operator_type = operator_type
        self.parent: Optional["Node"] = None
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None

        if self.value is not None:
            if self.value in BINARY_OPERATORS:
                self.operator_type = 2
            elif self.value in UNARY_OPERATORS:
                self.operator_type = 1
            elif (
                isinstance(self.value, (int, float)) or 
                self.value in NULLARY_VARIABLES):
                self.operator_type = 0
            else:
                raise ValueError(f"{self.value} is an invalid operator / variable")
        
        else:
            if self.operator_type == 2:
                self.value = np.random.choice(BINARY_OPERATORS)
            elif self.operator_type == 1:
                self.value = np.random.choice(UNARY_OPERATORS)
            elif self.operator_type == 0:
                randint = np.random.randint(0, 2)
                if  randint == 0:
                    self.value = np.random.choice(NULLARY_VARIABLES)
                else:
                    self.value = np.random.uniform(*NULLARY_PARAMS_INTERVAL)
            else:
                raise ValueError(f"{self.operator_type} is an invalid operator type")

    def set_child(self, child: "Node", left_or_right: str):
        setattr(self, left_or_right, child)
        setattr(getattr(self, left_or_right), "parent", self)
        setattr(getattr(self, left_or_right), "level", self.level+1)

    def to_leaf(self, value: Union[str, float]):
        if value in BINARY_OPERATORS + UNARY_OPERATORS:
            raise Exception("leaf node cannot be a unary or binary operator")
        self.value = value
        self.operator_type = 0
        self.left = None
        self.right = None

    def replace_with_child(self, left_or_right: str):
        child_node: "Node" = getattr(self, left_or_right)
        self.value = child_node.value
        self.operator_type = child_node.operator_type
        self.left = child_node.left
        self.right = child_node.right
        Node.reset_levels(self, level=self.level)

    @staticmethod
    def reset_levels(node: "Node", level: int=1):
        if level < 1: raise ValueError("level of node cannot be less than 1")
        node.level = level
        if node.left is not None:
            Node.reset_levels(node.left, level + 1)
        if node.right is not None:
            Node.reset_levels(node.right, level + 1)

    def get_progenitor(self):
        if self.parent is None:
            return self
        return self.parent.get_progenitor()

    def __str__(self) -> str:
        if self.level == 1:
            ret = f"root: {self.value}\n"
        else:
            ret = f"{self.value}\n"
        for child in ["left", "right"]:
            child_node = getattr(self, child)
            if child_node is not None:
                ret += "\t"*(child_node.level-1) + f"{child}: {child_node.__str__()}"
        return ret
    
    def __repr__(self) -> str:
        return self.__str__()