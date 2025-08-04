import torch
import torch.nn as nn
import numpy as np

class HermitePolynomials(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(HermitePolynomials, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.degree = degree
        self.hermite_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.hermite_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        a, b, c = x.shape
        x = torch.reshape(x, (-1, self.input_dim))
        x = torch.tanh(x)
        hermite = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            hermite[:, :, 1] = 2 * x
        for i in range(2, self.degree + 1):
            hermite[:, :, i] = 2 * x * hermite[:, :, i - 1].clone() - 2 * (i - 1) * hermite[:, :, i - 2].clone()
        y = torch.einsum('bid,iod->bo', hermite, self.hermite_coeffs)
        y = y.view(a, b, self.out_dim)

        return y
