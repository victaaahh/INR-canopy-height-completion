import torch
import torch.nn as nn


class PSNR(nn.Module):
    def __init__(self, data_range: float):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.data_range = data_range

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse(predictions, target)
        psnr = 20 * torch.log10(torch.Tensor(self.data_range)) - 10 * torch.log10(mse)
        return psnr


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(predictions, target))


class ME(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(predictions - target)


class R2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(target)
        ss_res = torch.sum((target - predictions) ** 2)
        ss_tot = torch.sum((target - mean) ** 2)
        return 1.0 - ss_res / ss_tot
