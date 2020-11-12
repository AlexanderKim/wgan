from typing import List, Any

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import Trainer
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from discriminator import Discriminator
from faces_data_module import FacesDataModule
from generator import Generator


class WGAN(pl.LightningModule):
    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = BCEWithLogitsLoss()

    def forward(self, noise):
        return self.generator(noise)

    def train_generator(self, real):
        noise = self.generator.gen_noize(len(real), device=self.device)
        fake = self.generator(noise)
        fake_pred = self.discriminator(fake)
        fake_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))

        self.log_dict({"g_loss": fake_loss})

        return fake_loss

    def train_discriminator(self, real):
        real_pred = self.discriminator(real)
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))

        noise = self.generator.gen_noize(len(real), device=self.device)
        fake = self.generator(noise)
        fake_pred = self.discriminator(fake.detach())
        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))

        disc_loss = (fake_loss + real_loss) / 2

        self.log_dict({"d_loss": disc_loss})

        return disc_loss

    def training_step(self, batch, batch_idx, optimizer_idx):

        real, _ = batch

        if optimizer_idx == 0:
            return self.train_generator(real)
        if optimizer_idx == 1:
            return self.train_discriminator(real)

    def configure_optimizers(self):
        optimizer_gen = Adam(params=self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_disc = Adam(params=self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return optimizer_gen, optimizer_disc

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        noise = self.generator.gen_noize()
        fake_pred = self.generator(noise)
        img_grid = torchvision.utils.make_grid(fake_pred)
        self.logger.experiment.add_image('generated_images', img_grid, self.current_epoch)


if __name__ == "__main__":
    data_module = FacesDataModule()
    wgan = WGAN(generator=Generator(), discriminator=Discriminator())

    trainer = Trainer()
    trainer.fit(wgan, data_module)