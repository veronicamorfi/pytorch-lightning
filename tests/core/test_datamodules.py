# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict
from unittest import mock
from unittest.mock import call, PropertyMock

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests.helpers import BoringDataModule, BoringModel, RandomDataset
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel
from tests.helpers.utils import reset_seed


@mock.patch("pytorch_lightning.trainer.trainer.Trainer.node_rank", new_callable=PropertyMock)
@mock.patch("pytorch_lightning.trainer.trainer.Trainer.local_rank", new_callable=PropertyMock)
def test_can_prepare_data(local_rank, node_rank):
    dm = BoringDataModule()
    trainer = Trainer()
    trainer.datamodule = dm

    # 1 no DM
    # prepare_data_per_node = True
    # local rank = 0   (True)
    dm.random_full = None
    dm._has_prepared_data = False
    local_rank.return_value = 0
    assert trainer.local_rank == 0

    trainer._data_connector.prepare_data()
    assert dm.random_full is not None

    # local rank = 1   (False)
    dm.random_full = None
    dm._has_prepared_data = False
    local_rank.return_value = 1
    assert trainer.local_rank == 1

    trainer._data_connector.prepare_data()
    assert dm.random_full is None

    # prepare_data_per_node = False (prepare across all nodes)
    # global rank = 0   (True)
    dm.random_full = None
    dm._has_prepared_data = False
    dm.prepare_data_per_node = False
    node_rank.return_value = 0
    local_rank.return_value = 0

    trainer._data_connector.prepare_data()
    assert dm.random_full is not None

    # global rank = 1   (False)
    dm.random_full = None
    dm._has_prepared_data = False
    node_rank.return_value = 1
    local_rank.return_value = 0

    trainer._data_connector.prepare_data()
    assert dm.random_full is None

    node_rank.return_value = 0
    local_rank.return_value = 1

    trainer._data_connector.prepare_data()
    assert dm.random_full is None

    # 2 dm
    # prepar per node = True
    # local rank = 0 (True)
    dm.prepare_data_per_node = True
    local_rank.return_value = 0

    with mock.patch.object(trainer.datamodule, "prepare_data") as dm_mock:
        # is_overridden prepare data = True
        # has been called
        # False
        dm._has_prepared_data = True
        trainer._data_connector.prepare_data()
        dm_mock.assert_not_called()

        # has not been called
        # True
        dm._has_prepared_data = False
        trainer._data_connector.prepare_data()
        dm_mock.assert_called_once()


def test_hooks_no_recursion_error():
    # hooks were appended in cascade every tine a new data module was instantiated leading to a recursion error.
    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/3652
    class DummyDM(LightningDataModule):
        def setup(self, *args, **kwargs):
            pass

        def prepare_data(self, *args, **kwargs):
            pass

    for i in range(1005):
        dm = DummyDM()
        dm.setup()
        dm.prepare_data()


def test_helper_boringdatamodule():
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup()


def test_helper_boringdatamodule_with_verbose_setup():
    dm = BoringDataModule()
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")


def test_data_hooks_called():
    dm = BoringDataModule()
    assert not dm.has_prepared_data
    assert not dm.has_setup_fit
    assert not dm.has_setup_test
    assert not dm.has_setup_validate
    assert not dm.has_setup_predict
    assert not dm.has_teardown_fit
    assert not dm.has_teardown_test
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_predict

    dm.prepare_data()
    assert dm.has_prepared_data
    assert not dm.has_setup_fit
    assert not dm.has_setup_test
    assert not dm.has_setup_validate
    assert not dm.has_setup_predict
    assert not dm.has_teardown_fit
    assert not dm.has_teardown_test
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_predict

    dm.setup()
    assert dm.has_prepared_data
    assert dm.has_setup_fit
    assert dm.has_setup_test
    assert dm.has_setup_validate
    assert not dm.has_setup_predict
    assert not dm.has_teardown_fit
    assert not dm.has_teardown_test
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_predict

    dm.teardown()
    assert dm.has_prepared_data
    assert dm.has_setup_fit
    assert dm.has_setup_test
    assert dm.has_setup_validate
    assert not dm.has_setup_predict
    assert dm.has_teardown_fit
    assert dm.has_teardown_test
    assert dm.has_teardown_validate
    assert not dm.has_teardown_predict


@pytest.mark.parametrize("use_kwarg", (False, True))
def test_data_hooks_called_verbose(use_kwarg):
    dm = BoringDataModule()
    dm.prepare_data()
    assert not dm.has_setup_fit
    assert not dm.has_setup_test
    assert not dm.has_setup_validate
    assert not dm.has_setup_predict
    assert not dm.has_teardown_fit
    assert not dm.has_teardown_test
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_predict

    dm.setup(stage="fit") if use_kwarg else dm.setup("fit")
    assert dm.has_setup_fit
    assert not dm.has_setup_validate
    assert not dm.has_setup_test
    assert not dm.has_setup_predict

    dm.setup(stage="validate") if use_kwarg else dm.setup("validate")
    assert dm.has_setup_fit
    assert dm.has_setup_validate
    assert not dm.has_setup_test
    assert not dm.has_setup_predict

    dm.setup(stage="test") if use_kwarg else dm.setup("test")
    assert dm.has_setup_fit
    assert dm.has_setup_validate
    assert dm.has_setup_test
    assert not dm.has_setup_predict

    dm.setup(stage="predict") if use_kwarg else dm.setup("predict")
    assert dm.has_setup_fit
    assert dm.has_setup_validate
    assert dm.has_setup_test
    assert dm.has_setup_predict

    dm.teardown(stage="fit") if use_kwarg else dm.teardown("fit")
    assert dm.has_teardown_fit
    assert not dm.has_teardown_validate
    assert not dm.has_teardown_test
    assert not dm.has_teardown_predict

    dm.teardown(stage="validate") if use_kwarg else dm.teardown("validate")
    assert dm.has_teardown_fit
    assert dm.has_teardown_validate
    assert not dm.has_teardown_test
    assert not dm.has_teardown_predict

    dm.teardown(stage="test") if use_kwarg else dm.teardown("test")
    assert dm.has_teardown_fit
    assert dm.has_teardown_validate
    assert dm.has_teardown_test
    assert not dm.has_teardown_predict

    dm.teardown(stage="predict") if use_kwarg else dm.teardown("predict")
    assert dm.has_teardown_fit
    assert dm.has_teardown_validate
    assert dm.has_teardown_test
    assert dm.has_teardown_predict


def test_dm_add_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = BoringDataModule.add_argparse_args(parser)
    args = parser.parse_args(["--data_dir", str(tmpdir)])
    assert args.data_dir == str(tmpdir)


def test_dm_init_from_argparse_args(tmpdir):
    parser = ArgumentParser()
    parser = BoringDataModule.add_argparse_args(parser)
    args = parser.parse_args(["--data_dir", str(tmpdir)])
    dm = BoringDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()
    assert dm.data_dir == args.data_dir == str(tmpdir)


def test_dm_pickle_after_init():
    dm = BoringDataModule()
    pickle.dumps(dm)


def test_train_loop_only(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, enable_model_summary=False)

    # fit model
    trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.callback_metrics["train_loss"] < 1.0


def test_train_val_loop_only(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, enable_model_summary=False)

    # fit model
    trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.callback_metrics["train_loss"] < 1.0


def test_dm_checkpoint_save_and_load(tmpdir):
    class CustomBoringModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            self.log("early_stop_on", out["x"])
            return out

    class CustomBoringDataModule(BoringDataModule):
        def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            checkpoint[self.__class__.__name__] = self.__class__.__name__

        def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            self.checkpoint_state = checkpoint.get(self.__class__.__name__)

    reset_seed()
    dm = CustomBoringDataModule()
    model = CustomBoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        enable_model_summary=False,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor="early_stop_on")],
    )

    # fit model
    trainer.fit(model, datamodule=dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]
    checkpoint = torch.load(checkpoint_path)
    assert dm.__class__.__name__ in checkpoint
    assert checkpoint[dm.__class__.__name__] == dm.__class__.__name__

    for trainer_fn in TrainerFn:
        trainer.state.fn = trainer_fn
        with mock.patch.object(dm, "on_load_checkpoint") as dm_mock:
            trainer._restore_modules_and_callbacks(checkpoint_path)
            dm_mock.assert_called_once()


def test_full_loop(tmpdir):
    reset_seed()

    dm = ClassifDataModule()
    model = ClassificationModel()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, enable_model_summary=False, deterministic=True)

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # validate
    result = trainer.validate(model, dm)
    assert dm.trainer is not None
    assert result[0]["val_acc"] > 0.7

    # test
    result = trainer.test(model, dm)
    assert dm.trainer is not None
    assert result[0]["test_acc"] > 0.6


@RunIf(min_gpus=1)
@mock.patch("pytorch_lightning.accelerators.accelerator.Accelerator.lightning_module", new_callable=PropertyMock)
def test_dm_apply_batch_transfer_handler(get_module_mock):
    expected_device = torch.device("cuda", 0)

    class CustomBatch:
        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestDM(LightningDataModule):
        rank = 0
        transfer_batch_to_device_hook_rank = None
        on_before_batch_transfer_hook_rank = None
        on_after_batch_transfer_hook_rank = None

        def on_before_batch_transfer(self, batch, dataloader_idx):
            assert dataloader_idx == 0
            self.on_before_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.samples += 1
            return batch

        def on_after_batch_transfer(self, batch, dataloader_idx):
            assert dataloader_idx == 0
            assert batch.samples.device == batch.targets.device == expected_device
            self.on_after_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.targets *= 2
            return batch

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            assert dataloader_idx == 0
            self.transfer_batch_to_device_hook_rank = self.rank
            self.rank += 1
            batch.samples = batch.samples.to(device)
            batch.targets = batch.targets.to(device)
            return batch

    dm = CurrentTestDM()
    model = BoringModel()

    batch = CustomBatch((torch.zeros(5, 32), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer(gpus=1)
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead
    get_module_mock.return_value = model
    if is_overridden("transfer_batch_to_device", dm):
        model.transfer_batch_to_device = dm.transfer_batch_to_device

    model.on_before_batch_transfer = dm.on_before_batch_transfer
    model.transfer_batch_to_device = dm.transfer_batch_to_device
    model.on_after_batch_transfer = dm.on_after_batch_transfer

    batch_gpu = trainer.accelerator.batch_to_device(batch, expected_device)

    assert dm.on_before_batch_transfer_hook_rank == 0
    assert dm.transfer_batch_to_device_hook_rank == 1
    assert dm.on_after_batch_transfer_hook_rank == 2
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected_device
    assert torch.allclose(batch_gpu.samples.cpu(), torch.ones(5, 32))
    assert torch.allclose(batch_gpu.targets.cpu(), torch.ones(5, 1, dtype=torch.long) * 2)


def test_dm_reload_dataloaders_every_n_epochs(tmpdir):
    """Test datamodule, where trainer argument reload_dataloaders_every_n_epochs is set to a non negative
    integer."""

    class CustomBoringDataModule(BoringDataModule):
        def __init__(self):
            super().__init__()
            self._epochs_called_for = []

        def train_dataloader(self):
            assert self.trainer.current_epoch not in self._epochs_called_for
            self._epochs_called_for.append(self.trainer.current_epoch)
            return super().train_dataloader()

    dm = CustomBoringDataModule()
    model = BoringModel()

    model.validation_step = None
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = None
    model.test_step_end = None
    model.test_epoch_end = None

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=3, limit_train_batches=2, reload_dataloaders_every_n_epochs=2)
    trainer.fit(model, dm)


class DummyDS(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return 1

    def __len__(self):
        return 100


class DummyIDS(torch.utils.data.IterableDataset):
    def __iter__(self):
        yield 1


@pytest.mark.parametrize("iterable", (False, True))
def test_dm_init_from_datasets_dataloaders(iterable):
    ds = DummyIDS if iterable else DummyDS

    train_ds = ds()
    dm = LightningDataModule.from_datasets(train_ds, batch_size=4, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.train_dataloader()
        dl_mock.assert_called_once_with(train_ds, batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True)
    with pytest.raises(NotImplementedError):
        _ = dm.val_dataloader()
    with pytest.raises(NotImplementedError):
        _ = dm.test_dataloader()

    train_ds_sequence = [ds(), ds()]
    dm = LightningDataModule.from_datasets(train_ds_sequence, batch_size=4, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.train_dataloader()
        dl_mock.assert_has_calls(
            [
                call(train_ds_sequence[0], batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True),
                call(train_ds_sequence[1], batch_size=4, shuffle=not iterable, num_workers=0, pin_memory=True),
            ]
        )
    with pytest.raises(NotImplementedError):
        _ = dm.val_dataloader()
    with pytest.raises(NotImplementedError):
        _ = dm.test_dataloader()

    valid_ds = ds()
    test_ds = ds()
    dm = LightningDataModule.from_datasets(val_dataset=valid_ds, test_dataset=test_ds, batch_size=2, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.val_dataloader()
        dl_mock.assert_called_with(valid_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
        dm.test_dataloader()
        dl_mock.assert_called_with(test_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    with pytest.raises(NotImplementedError):
        _ = dm.train_dataloader()

    valid_dss = [ds(), ds()]
    test_dss = [ds(), ds()]
    dm = LightningDataModule.from_datasets(train_ds, valid_dss, test_dss, batch_size=4, num_workers=0)
    with mock.patch("pytorch_lightning.core.datamodule.DataLoader") as dl_mock:
        dm.val_dataloader()
        dm.test_dataloader()
        dl_mock.assert_has_calls(
            [
                call(valid_dss[0], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
                call(valid_dss[1], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
                call(test_dss[0], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
                call(test_dss[1], batch_size=4, shuffle=False, num_workers=0, pin_memory=True),
            ]
        )


# all args
class DataModuleWithHparams_0(LightningDataModule):
    def __init__(self, arg0, arg1, kwarg0=None):
        super().__init__()
        self.save_hyperparameters()


# single arg
class DataModuleWithHparams_1(LightningDataModule):
    def __init__(self, arg0, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(arg0)


def test_hyperparameters_saving():
    data = DataModuleWithHparams_0(10, "foo", kwarg0="bar")
    assert data.hparams == AttributeDict({"arg0": 10, "arg1": "foo", "kwarg0": "bar"})

    data = DataModuleWithHparams_1(Namespace(**{"hello": "world"}), "foo", kwarg0="bar")
    assert data.hparams == AttributeDict({"hello": "world"})

    data = DataModuleWithHparams_1({"hello": "world"}, "foo", kwarg0="bar")
    assert data.hparams == AttributeDict({"hello": "world"})

    data = DataModuleWithHparams_1(OmegaConf.create({"hello": "world"}), "foo", kwarg0="bar")
    assert data.hparams == OmegaConf.create({"hello": "world"})


def test_define_as_dataclass():
    # makes sure that no functionality is broken and the user can still manually make
    # super().__init__ call with parameters
    # also tests all the dataclass features that can be enabled without breaking anything
    @dataclass(init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
    class BoringDataModule1(LightningDataModule):
        batch_size: int
        dims: int = 2

        def train_dataloader(self):
            return DataLoader(torch.randn(self.batch_size * 2, 10), batch_size=self.batch_size)

    # asserts for the different dunder methods added by dataclass, when __init__ is implemented, i.e.
    # __repr__, __eq__, __lt__, __le__, etc.
    assert BoringDataModule1(batch_size=64).dims == 2
    assert BoringDataModule1(batch_size=32)
    assert len(BoringDataModule1(batch_size=32)) == 2
    assert hasattr(BoringDataModule1, "__repr__")
    assert BoringDataModule1(batch_size=32) == BoringDataModule1(batch_size=32)

    # asserts inherent calling of super().__init__ in case user doesn't make the call
    @dataclass
    class BoringDataModule2(LightningDataModule):
        batch_size: int

    # asserts for the different dunder methods added by dataclass, when super class is inherently initialized, i.e.
    # __init__, __repr__, __eq__, __lt__, __le__, etc.
    assert BoringDataModule2(batch_size=32) is not None
    assert BoringDataModule2(batch_size=32).batch_size == 32
    assert len(BoringDataModule2(batch_size=32)) == 0
    assert hasattr(BoringDataModule2, "__repr__")
    assert BoringDataModule2(batch_size=32).prepare_data() is None
    assert BoringDataModule2(batch_size=32) == BoringDataModule2(batch_size=32)

    # checking for all the different multilevel inhertiance scenarios, for init call on LightningDataModule
    @dataclass
    class BoringModuleBase1(LightningDataModule):
        num_features: int

    class BoringModuleBase2(LightningDataModule):
        def __init__(self, num_features: int):
            self.num_features = num_features

    @dataclass
    class BoringModuleDerived1(BoringModuleBase1):
        ...

    class BoringModuleDerived2(BoringModuleBase1):
        def __init__(self):
            ...

    @dataclass
    class BoringModuleDerived3(BoringModuleBase2):
        ...

    class BoringModuleDerived4(BoringModuleBase2):
        def __init__(self):
            ...

    assert hasattr(BoringModuleDerived1(num_features=2), "_has_prepared_data")
    assert hasattr(BoringModuleDerived2(), "_has_prepared_data")
    assert hasattr(BoringModuleDerived3(), "_has_prepared_data")
    assert hasattr(BoringModuleDerived4(), "_has_prepared_data")


def test_inconsistent_prepare_data_per_node(tmpdir):
    with pytest.raises(MisconfigurationException, match="Inconsistent settings found for `prepare_data_per_node`."):
        model = BoringModel()
        dm = BoringDataModule()
        trainer = Trainer(prepare_data_per_node=False)
        trainer.model = model
        trainer.datamodule = dm
        trainer._data_connector.prepare_data()


DATALOADER = DataLoader(RandomDataset(1, 32))


@pytest.mark.parametrize("method_name", ["train_dataloader", "val_dataloader", "test_dataloader", "predict_dataloader"])
@pytest.mark.parametrize(
    ["dataloader", "expected"],
    [
        [DATALOADER, 32],
        [[DATALOADER, DATALOADER], 64],
        [[[DATALOADER], [DATALOADER, DATALOADER]], 96],
        [[{"foo": DATALOADER}, {"foo": DATALOADER, "bar": DATALOADER}], 96],
        [{"foo": DATALOADER, "bar": DATALOADER}, 64],
        [{"foo": {"foo": DATALOADER}, "bar": {"foo": DATALOADER, "bar": DATALOADER}}, 96],
        [{"foo": [DATALOADER], "bar": [DATALOADER, DATALOADER]}, 96],
        [CombinedLoader({"foo": DATALOADER, "bar": DATALOADER}), 64],
    ],
)
def test_len_different_types(method_name, dataloader, expected):
    dm = LightningDataModule()
    setattr(dm, method_name, lambda: dataloader)
    assert len(dm) == expected


@pytest.mark.parametrize("method_name", ["train_dataloader", "val_dataloader", "test_dataloader", "predict_dataloader"])
def test_len_dataloader_no_len(method_name):
    class CustomNotImplementedErrorDataloader(DataLoader):
        def __len__(self):
            raise NotImplementedError

    dataloader = CustomNotImplementedErrorDataloader(RandomDataset(1, 32))
    dm = LightningDataModule()
    setattr(dm, method_name, lambda: dataloader)
    with pytest.warns(UserWarning, match=f"The number of batches for a dataloader in `{method_name}` is counted as 0"):
        assert len(dm) == 0


def test_len_all_dataloader_methods_implemented():
    class BoringDataModule(LightningDataModule):
        def __init__(self, dataloader):
            super().__init__()
            self.dataloader = dataloader

        def train_dataloader(self):
            return {"foo": self.dataloader, "bar": self.dataloader}

        def val_dataloader(self):
            return self.dataloader

        def test_dataloader(self):
            return [self.dataloader]

        def predict_dataloader(self):
            return [self.dataloader, self.dataloader]

    dm = BoringDataModule(DATALOADER)

    # 6 dataloaders each producing 32 batches: 6 * 32 = 192
    assert len(dm) == 192


def test_len_no_dataloader_methods_implemented():
    dm = LightningDataModule()
    with pytest.warns(UserWarning, match="You datamodule does not have any valid dataloader"):
        assert len(dm) == 0

    dm.train_dataloader = None
    dm.val_dataloader = None
    dm.test_dataloader = None
    dm.predict_dataloader = None
    with pytest.warns(UserWarning, match="You datamodule does not have any valid dataloader"):
        assert len(dm) == 0
