"""Native-finishing-policy controls using A0–A3 unchanged architecture."""
from yolox.core.fourier_trainer import FourierTrainer
from yolox.core.a4_schedule import before_epoch_with_native_finish


class A4MatchedControlTrainer(FourierTrainer):
    def before_epoch(self):
        return before_epoch_with_native_finish(self)

    def before_train(self):
        super().before_train()
        if self.rank==0:
            import hashlib
            import json
            from pathlib import Path
            src=Path(__file__).with_name('a4_schedule.py')
            dest=Path(self.file_name)/'a4_matched_policy.json'
            dest.write_text(json.dumps({
                'policy':'same native-finish Mosaic-off/L1 schedule as A4',
                'seed':self.exp.seed,'variant':self.exp.fourier_variant,
                'batch_size':self.args.batch_size,
                'scheduler_horizon':self.max_epoch,
                'policy_sha256':hashlib.sha256(src.read_bytes()).hexdigest()
            },indent=2),encoding='utf-8')
