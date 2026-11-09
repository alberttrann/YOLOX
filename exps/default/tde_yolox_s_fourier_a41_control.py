"""A4.1-R: fixed random orthogonal operator-matched non-DCT control."""
from tde_yolox_s_fourier_a41 import Exp as A41Exp


class Exp(A41Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = 'tdevnext3_a41_orthogonal_grounded'
        self.fourier_variant = 'a41_control'
        self.a4_filter_basis = 'orthogonal_control'
