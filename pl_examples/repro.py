import os
import torch
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DeepSpeedPlugin


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run(resume=False):
    os.environ["PL_FAULT_TOLERANT_TRAINING"] = "1"
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(320, 64), batch_size=2)

    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:02d}",
    )
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        gpus=2,
        num_sanity_val_steps=0,
        precision=16,
        accelerator="ddp",
        max_epochs=2,
        plugins=[DeepSpeedPlugin(stage=2)],
        weights_summary=None,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint="checkpoints/epoch=01.ckpt" if resume else None,
    )
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume)
