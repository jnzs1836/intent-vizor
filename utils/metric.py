class Metric(object):
    def __init__(self, name):
        self.name = name
        self.values = []

    def add_value(self, value, weight=1):
        self.values.append(weight * value)

    def avg(self):
        if len(self.values) == 0:
            return 0
        return sum(self.values) / len(self.values)


class Meter:
    def __init__(self, name):
        self.name = name
        self.values = []

    def add_value(self, value, weight=1):
        self.values.append(weight * value)

    def avg(self):
        if len(self.values) == 0:
            return 0
        return sum(self.values) / len(self.values)
