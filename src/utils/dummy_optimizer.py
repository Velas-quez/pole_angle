import torch
from torch.optim.optimizer import Optimizer


class DummyOptimizer(Optimizer):
    def __init__(self, params, lr=0.0, **kwargs):
        # Criamos um tensor inútil com requires_grad=True
        dummy_param = torch.nn.Parameter(torch.zeros(1))

        # O dicionário 'defaults' é necessário para a classe base configurar o param_groups
        defaults = {"lr": lr}

        # Inicializa a classe base com o parâmetro falso
        super().__init__([dummy_param], defaults)

    def step(self, closure=None):
        pass
