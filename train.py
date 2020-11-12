from pytorch_lightning import Trainer

from discriminator import Discriminator
from faces_data_module import FacesDataModule
from generator import Generator
from wgan import WGAN

if __name__ == "__main__":
    data_module = FacesDataModule()
    wgan = WGAN(generator=Generator(), discriminator=Discriminator())

    trainer = Trainer(automatic_optimization=False)
    trainer.fit(wgan, data_module)