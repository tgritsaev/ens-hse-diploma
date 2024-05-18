from dataclasses import dataclass

import random

import torch
import torch.nn as nn


def get_weights_as_tensor(model, detach):
    if detach:
        params = [p.data.detach().ravel() for p in model.parameters()]
    else:
        params = [p.data.ravel() for p in model.parameters()]
    return torch.cat(params)

@dataclass
class WeightsOrthogonalizerVector:
    weights_vector: torch.Tensor
    weights_norm: float
    
    def __iter__(self):
        return iter((self.weights_vector, self.weights_norm))

class WeightsOrthogonalizer(nn.Module):
    
    def __init__(
        self, 
        device,
        # use_weights_orthogonalizer_first_n_epochs=100, 
        **kwargs,
    ):
        super().__init__()
        
        self.device = device
        self.vector_first = None
        self.orthogonalizer_vectors = []
        # self.use_first_n_epochs = use_weights_orthogonalizer_first_n_epochs
        # self.epoch_num = 0
        
    # def new_epoch(self):
    #     self.epoch_num += 1
        
    def insert_model(self, model):
        if self.vector_first is None:
            self.vector_first = get_weights_as_tensor(model, True)
        else:
            # vector_first_current = get_weights_as_tensor(model, True) - self.vector_first
            # self.orthogonalizer_vectors.append(
            #     WeightsOrthogonalizerVector(vector_first_current, torch.linalg.norm(vector_first_current))
            # )
            current_weights = get_weights_as_tensor(model, True)
            self.orthogonalizer_vectors.append(
                WeightsOrthogonalizerVector(current_weights, torch.linalg.norm(current_weights))
            )
        # self.epoch_num = 0
        
    def forward(self, model):
        # \or self.epoch_num > self.use_first_n_epochs:
        if len(self.orthogonalizer_vectors) == 0:
            return torch.tensor([0.0], device=self.device)
        
        orthogonalizer_vector, orthogonalizer_vector_norm = random.choice(self.orthogonalizer_vectors)
        # vector_first_current = get_weights_as_tensor(model, False) - self.vector_first
        current_weights = get_weights_as_tensor(model, False)
        
        get_cosine_similarity = lambda a, b, b_norm: torch.sum(a * b) / (torch.linalg.norm(a) * b_norm)
        return torch.square(
            get_cosine_similarity(current_weights, orthogonalizer_vector, orthogonalizer_vector_norm)
        )