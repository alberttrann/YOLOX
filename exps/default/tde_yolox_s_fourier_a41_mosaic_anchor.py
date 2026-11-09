"""A4 architecture under A4.1's corrected REAL Mosaic/terminal data policy.

Needed to separate data-loader repair from the bundled A4.1 changes.
Only compare this against historical A4 to estimate the data-policy effect;
A4.1 DCT vs this anchor compares models under matched real Mosaic.
"""
from tde_yolox_s_fourier_a41 import Exp as A41Exp
from tde_yolox_s_fourier_a4 import Exp as A4Exp


class Exp(A41Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = 'tdevnext3_a41_mosaic_a4_anchor'
        self.fourier_variant = 'a41_mosaic_a4'
        self.a41_architecture = 'a4_original_modules_with_real_mosaic'
        self.a41_proto_weight = 1.0  # A4 effective coefficient, not A4.1's 0.05

    def get_model(self):
        # Calls A4's exact preexisting model build (old SPI/Tribrid/Fourier/CE).
        # Do not route through A41Exp.get_model(): this is the anchor treatment.
        return A4Exp.get_model(self)
