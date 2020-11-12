from typing import List, Any

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from discriminator import Discriminator
from generator import Generator


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class WGAN(pl.LightningModule):
    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()

        self.generator = generator.apply(weights_init)
        self.discriminator = discriminator.apply(weights_init)

    def forward(self, noise):
        return self.generator(noise)

    def get_gradient(self, real, fake, epsilon):
        mixed_images = real * epsilon + fake * (1 - epsilon)
        mixed_scores = self.discriminator(mixed_images)

        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]

        return gradient

    def gradient_penalty(self, gradient):
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)

        penalty = torch.mean(torch.pow(gradient_norm - 1, 2))

        return penalty

    def gen_loss(self, fake_pred):
        return - torch.mean(fake_pred)

    def disc_loss(self, fake_pred, real_pred, gradient_penalty, c_lambda):
        return torch.mean(fake_pred) - torch.mean(real_pred) + c_lambda * gradient_penalty

    def train_generator(self, real, optimizer):
        noise = self.generator.gen_noize(len(real), device=self.device)
        fake = self.generator(noise)
        fake_pred = self.discriminator(fake)
        fake_loss = self.gen_loss(fake_pred)

        self.manual_backward(fake_loss, optimizer)
        optimizer.step()

        self.log_dict({"g_loss": fake_loss})

        return fake_loss

    def train_discriminator(self, real, optimizer, repeats=5, c_lambda=10):
        mean_disc_loss = 0

        for _ in range(repeats):
            optimizer.zero_grad()

            real_pred = self.discriminator(real)

            noise = self.generator.gen_noize(len(real), device=self.device)
            fake = self.generator(noise)
            fake_pred = self.discriminator(fake.detach())

            epsilon = torch.rand(len(real), 1, 1, 1, requires_grad=True, device=self.device)
            gradient = self.get_gradient(real, fake.detach(), epsilon)
            gradient_penalty = self.gradient_penalty(gradient)
            disc_loss = self.disc_loss(fake_pred, real_pred, gradient_penalty, c_lambda)

            mean_disc_loss += disc_loss.item() / repeats

            self.manual_backward(disc_loss, optimizer, retain_graph=True)
            optimizer.step()

        self.log_dict({"d_loss": mean_disc_loss})

        return mean_disc_loss

    def training_step(self, batch, batch_idx, optimizer_idx):

        real, _ = batch

        self.train_generator(real, self.optimizers()[0])
        self.train_discriminator(real, self.optimizers()[1])

        # if optimizer_idx == 0:
        #     return self.train_generator(real, optimizer)
        # if optimizer_idx == 1:
        #     return self.train_discriminator(real, optimizer)

    def configure_optimizers(self):
        optimizer_gen = Adam(params=self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_disc = Adam(params=self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return [optimizer_gen, optimizer_disc], []

    def on_epoch_end(self):
        noise = self.generator.gen_noize(device=self.device)
        fake_pred = self.generator(noise)
        img_grid = torchvision.utils.make_grid(fake_pred)
        self.logger.experiment.add_image('generated_images', img_grid, self.current_epoch)

