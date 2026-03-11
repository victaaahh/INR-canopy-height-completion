from torch.utils.data import Dataset


class SingleImage(Dataset):
    def __init__(self, image):
        self.image = image
