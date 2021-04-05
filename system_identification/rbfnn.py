

class RadialBasisFunctionNeuralNetwork:
    def __init__(self):
        self.IW = None
        self.LW = None
        self.b = None
        self.range = None
        self.trainParam = None
        self.epochs = None
        self.goal = None
        self.min_grad = None
        self.mu = None

        self.name = "None"
        self.trainFunct = None
        self.trainAlg = None
