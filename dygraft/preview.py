"""Runtime helpers extracted from the experiment implementation."""
from __future__ import annotations

import torch


class PreviewModel(torch.nn.Module):
    """Expose the latest denoised numeric preview and categorical prediction."""

    def __init__(self, base, category_sizes):
        super().__init__()
        self.base = base
        self.category_sizes = [int(value) for value in category_sizes]
        self.numeric = None
        self.codes = None

    def predicted_codes(self, x_cat, logits):
        current_chunks = torch.split(x_cat, [size + 1 for size in self.category_sizes], dim=-1)
        logit_chunks = torch.split(logits, [size + 1 for size in self.category_sizes], dim=-1)
        codes = []
        for size, current, prediction in zip(self.category_sizes, current_chunks, logit_chunks):
            current_code = current.argmax(dim=-1)
            predicted = prediction[:, :size].argmax(dim=-1)
            codes.append(torch.where(current_code.eq(size), predicted, current_code))
        return torch.stack(codes, dim=1)

    def forward(self, x_num, x_cat, t, sigma=None):
        output = self.base(x_num, x_cat, t, sigma=sigma)
        self.numeric = output[0].detach()
        self.codes = self.predicted_codes(x_cat, output[1]).detach()
        return output
