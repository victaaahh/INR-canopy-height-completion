from abc import ABC, abstractmethod

from torch import Tensor, nn
from typing_extensions import override


class INR(nn.Module, ABC):
    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def first_layer(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def middle_layers(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def last_layer(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def guide_head(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    @override
    def forward(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        pass
