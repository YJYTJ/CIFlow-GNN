import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import InstanceNorm




def get_preds(logits):
    
    preds = logits.argmax(dim=1).float()
    
    return preds


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs

