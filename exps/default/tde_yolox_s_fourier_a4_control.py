"""A4-R: operator matched, fixed orthogonal non-DCT filter-bank control."""
from tde_yolox_s_fourier_a4 import Exp as A4Exp


class Exp(A4Exp):
    def __init__(self):
        super().__init__()
        self.exp_name='tdevnext3_a4_orthogonal_control'
        self.fourier_variant='a4_control'
        self.a4_filter_basis='orthogonal_control'
