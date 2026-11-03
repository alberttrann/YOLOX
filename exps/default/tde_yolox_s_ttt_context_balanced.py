"""vNext.2 TTT context fill + symmetric region-normalized reconstruction."""
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
        self.ttt_context_mode = 'context_balanced'
        self.exp_name = 'tde_yolox_s_vnext2_ttt_context_balanced'
