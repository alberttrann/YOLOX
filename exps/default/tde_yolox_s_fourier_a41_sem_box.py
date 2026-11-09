"""A4.1 training-only semantic auxiliary research: legacy."""
from tde_yolox_s_fourier_a41 import Exp as A41Exp
from yolox.models.a41_sem_detector import A41SemanticDetector

class Exp(A41Exp):
    def __init__(self):
        super().__init__()
        self.exp_name = 'tdevnext3_a41_sem_box'
        self.a41_sem_mode = 'legacy'
        self.a41_sem_loss_weight = 0.25
        self.a41_sem_grad_scale = 0.05
        self.a41_sem_sigma_divisor = 6.0
        self.a41_sem_harmonic_weight = 0.0
        self.enable_afford_aux = True
        self.a41_sem_aux_train_only = True

    def get_model(self):
        if getattr(self,'model',None) is not None:
            return self.model
        native = super().get_model()
        self.model = A41SemanticDetector(
            native.backbone,native.head,width=self.width,
            aux_mode=self.a41_sem_mode,
            sem_weight=self.a41_sem_loss_weight,
            backbone_grad_scale=self.a41_sem_grad_scale,
            sigma_divisor=self.a41_sem_sigma_divisor,
            harmonic_weight=self.a41_sem_harmonic_weight)
        self.model.max_epochs = int(self.max_epoch)
        return self.model
