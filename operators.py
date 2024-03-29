import operator

BINARY_OPERATORS_FUNC_DICT = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "pow": operator.pow
}
BINARY_OPERATORS = list(BINARY_OPERATORS_FUNC_DICT.keys())

UNARY_OPERATORS = ["exp", "log", "sin", "cos"]
NULLARY_VARIABLES = ["x"]
NULLARY_PARAMS_INTERVAL = -30.0, 30.0