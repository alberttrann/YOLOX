"""Isolated vNext.2 TTT contextual-reconstruction experiment (NOT vNext.3).

Use the same pretrained detector/parameters; only the TTT reconstruction rule
changes. Never resume a legacy vNext.2 optimizer state into this experiment.
"""
# YOLOX get_exp() commonly imports the file with exps/default on sys.path.
# Support that as well as ordinary package imports.
try:
    from tde_yolox_s import Exp as BaseExp
except ModuleNotFoundError as error:
    if error.name != 'tde_yolox_s':
        raise
    from exps.default.tde_yolox_s import Exp as BaseExp

from yolox.models.ttt_context_ablation import install_contextual_ttt


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.ttt_context_mode = 'context_mse'
        self.exp_name = 'tde_yolox_s_vnext2_ttt_context_mse'

    def get_model(self):
        model = super().get_model()
        return install_contextual_ttt(model, self.ttt_context_mode)
