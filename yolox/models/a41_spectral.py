"""A4.1 non-starved DCT and operator-matched non-DCT residual adapters."""
import torch
from torch import nn
from .a4_spectral import A4SpectralResidual


class A41SpectralResidual(A4SpectralResidual):
    """Safe low-energy nonzero projector initialization.

    Unlike A4's zero-projector scheme, the DCT band logits and mixer receive
    classification gradients from the first backward. Energy attenuation and
    exact same ON/OFF counterfactual interface are preserved. The orthogonal
    control shares the exact trainable parameter names and shapes.
    """
    def __init__(self, channels, basis='dct'):
        super().__init__(channels, basis=basis)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.project.weight, a=5 ** 0.5)
            self.project.weight.mul_(0.018)
            nn.init.zeros_(self.project.bias)
        self.initialization = 'small_nonzero_projector_0.018'
