import copy
import random

import torch
import torch.nn as nn


class PrevCELoss(nn.Module):
    def __init__(self, device, only_negative=False, use_first=False, **kwargs):
        super().__init__()

        self.regularizer_models = []
        self.only_negative = only_negative
        self.device = device
        self.use_first = use_first
        self.cnt = 0

    def copy_regularizer_model(self, regularizer_model):
        if self.use_first or self.cnt > 0:
            self.regularizer_models.append(copy.deepcopy(regularizer_model).to(self.device))
            self.regularizer_models[-1].eval()
        self.cnt += 1

    def random_model_forward(self, images):
        if len(self.regularizer_models) == 0:
            return None
        with torch.no_grad():
            model = random.choice(self.regularizer_models)
            return model(images, False)

    def forward(self, preds, prev_preds, labels):
        if prev_preds is None:
            return torch.tensor([0.0], device=self.device)
        
        if self.only_negative:
            prev_ce_mask = (labels != prev_preds.argmax(-1))
            return nn.functional.cross_entropy(preds[prev_ce_mask], prev_preds.argmax(-1)[prev_ce_mask])
        else:
            return nn.functional.cross_entropy(preds, prev_preds.argmax(-1))
