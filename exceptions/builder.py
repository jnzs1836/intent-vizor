class InvalidSolverException(Exception):
    def __init__(self, solver):
        message = "The solver specified {} is not valid!".format(solver)
        super().__init__(message)


class InvalidModelException(Exception):
    def __init__(self, model):
        message = "The model specified {} is not valid!".format(model)
        super().__init__(message)
