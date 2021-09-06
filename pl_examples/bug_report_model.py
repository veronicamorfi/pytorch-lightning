import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer


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
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    os.environ["PL_FAULT_TOLERANT_TRAINING"] = "1"
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2, num_workers=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2, num_workers=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2, num_workers=2)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        num_sanity_val_steps=0,
        max_epochs=2,
        weights_summary=None,
        # resume_from_checkpoint="/Users/adrian/repositories/pytorch-lightning/pl_examples/lightning_logs/version_43/checkpoints/epoch=2-step=95.ckpt",
        plugins="deepspeed",
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)

    print(trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    run()
