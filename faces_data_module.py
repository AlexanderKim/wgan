import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class FacesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 128, path: str = "data"):
        super().__init__()

        self.batch_size = batch_size
        self.path = path

        self.transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def prepare_data(self, stage=None):
        self.dataset = torchvision.datasets.ImageFolder(root=self.path, transform=self.transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size)
