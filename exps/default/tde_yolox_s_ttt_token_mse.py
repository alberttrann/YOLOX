"""vNext.2 token + original MSE with per-image first-order train/TTT semantics."""
# YOLOX get_exp() commonly imports the file with exps/default on sys.path.
# Support that as well as ordinary package imports.
try:
    from tde_yolox_s_ttt_context_ablation import Exp as BaseExp
except ModuleNotFoundError as error:
    if error.name != 'tde_yolox_s_ttt_context_ablation':
        raise
    from exps.default.tde_yolox_s_ttt_context_ablation import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.ttt_context_mode = 'token_mse'
        self.exp_name = 'tde_yolox_s_vnext2_ttt_token_mse'
