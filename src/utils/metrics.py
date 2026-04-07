import torch
import torch.nn as nn


class PSNR(nn.Module):
    def __init__(self, data_range: float):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.data_range = data_range

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse(predictions, target)
        psnr = 20 * torch.log10(torch.tensor(self.data_range)) - 10 * torch.log10(mse)
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


class SSIM(nn.Module):
    """Uses special case formula as described in
    https://en.wikipedia.org/wiki/Structural_similarity_index_measure.
    Computes for all pixels without knowing location, so equivalent to a
    single centered window of image size and uniform kernel."""

    def __init__(self, data_range, k_1, k_2):
        super().__init__()
        self.c_1 = (data_range * k_1) * (data_range * k_1)
        self.c_2 = (data_range * k_2) * (data_range * k_2)

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, channels = predictions.shape
        assert channels == 1, "SSIM is only implemented for 1 channel images"
        mu_x = predictions.mean()
        mu_y = target.mean()
        sigma2_x = predictions.var()
        sigma2_y = predictions.var()
        sigma_xy = torch.cov(torch.vstack([predictions.T, target.T]))
        numerator = (2 * mu_x * mu_y + self.c_1) * (2 * sigma_xy + self.c_2)
        denominator = (mu_x * mu_x + mu_y * mu_y + self.c_1) * (
            sigma2_x + sigma2_y + self.c_2
        )
        return numerator / denominator
