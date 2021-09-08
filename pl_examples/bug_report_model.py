import os
import shutil

import torch
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


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
        print("train_step")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("val_loss", loss)
        print("val_step")
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run(resume=False):
    os.environ["PL_FAULT_TOLERANT_TRAINING"] = "1"
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_dataloader = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()

    if not resume and os.path.exists("checkpoints") and os.path.isdir("checkpoints"):
        shutil.rmtree("checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:02d}",
    )
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        gpus=2,  # for hang, switch to GPU
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator="ddp",
        max_epochs=(1 if not resume else 2),
        weights_summary=None,
        callbacks=[checkpoint_callback],
        # checkpoint_callback=(not resume),
        resume_from_checkpoint=("checkpoints/epoch=00.ckpt" if resume else None),
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume)
