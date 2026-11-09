from tde_yolox_s_fourier import Exp as BaseExp
class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name='tdevnext3_a2_spectral'
        self.fourier_variant='spectral'
