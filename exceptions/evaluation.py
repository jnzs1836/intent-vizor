class InvalidEvaluationMethod(Exception):
    def __init__(self, method):
        message = "The input evaluation method {} is not valid!".format(method)
        super().__init__(message)


class InvalidEvaluationMetric(Exception):
    def __init__(self, method):
        message = "The input evaluation metric {} is not valid!".format(method)
        super().__init__(message)


class InExistentLossInEvaluation(Exception):
    def __init__(self, loss_name):
        message = "The specified loss {} is not valid!".format(loss_name)
        super().__init__(message)
