"""A4 loss accounting separates task loss from prototype-only supervision."""
from .fourier_detector import FourierDetector


class A4Detector(FourierDetector):
    def forward(self, x, targets=None):
        result=super().forward(x,targets)
        if self.training and isinstance(result,dict):
            aux=self.head.aux_memory_loss
            if aux is None: aux=result['total_loss']*0
            result['mem_loss']=aux
            result['det_loss']=(result['total_loss']-aux).detach()
            result['prototype_only_aux']=aux.detach()
        return result
