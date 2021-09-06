import os

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin
from tests.helpers import RandomDataset, BoringModel


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
        precision=16,
        gpus=1,
        # resume_from_checkpoint="/Users/adrian/repositories/pytorch-lightning/pl_examples/lightning_logs/version_43/checkpoints/epoch=2-step=95.ckpt",
        plugins=DeepSpeedPlugin(),
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)

    print(trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    run()
