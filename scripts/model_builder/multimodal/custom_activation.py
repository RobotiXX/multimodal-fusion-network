import torch
import torch.nn as nn




class CustomActivation(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.slope = torch.tensor(0.001)

    def forward(self, input):
        return torch.multiply(self.slope, input)