import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


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
    def __init__(
        self, data_range, k_1, k_2, gaussian_kernel, sigma, kernel_size, device
    ):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            k1=k_1,
            k2=k_2,
            gaussian_kernel=gaussian_kernel,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction="elementwise_mean",
            return_full_image=False,
            return_contrast_sensitivity=False,
        ).to(device)

    def forward(
        self, predicted_image: torch.Tensor, target_image: torch.Tensor
    ) -> torch.Tensor:
        h, w, c = target_image.shape
        reshaped_predicted_image = predicted_image.permute(2, 0, 1).view(1, c, h, w)
        reshaped_target_image = target_image.permute(2, 0, 1).view(1, c, h, w)
        return self.ssim(reshaped_predicted_image, reshaped_target_image)
