import torch

class SGD:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.state = {param: {"m": torch.zeros_like(param), "v": torch.zeros_like(param)} for param in self.parameters}
        self.t = 0