import os

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin
from tests.helpers import RandomDataset, BoringModel


class SimpleModel(BoringModel):
    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        if self.current_epoch == 0 and batch_idx == 1:
            raise RuntimeError("fault tolerant exception")


def run():
    os.environ["PL_FAULT_TOLERANT_TRAINING"] = "1"
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2, num_workers=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2, num_workers=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2, num_workers=2)

    model = BoringModel()
    model.training_epoch_end = None
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
        precision=16,
        gpus=2,
        # resume_from_checkpoint=".pl_auto_save.ckpt",
        plugins=DeepSpeedPlugin(),
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)

    print(trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    run()
