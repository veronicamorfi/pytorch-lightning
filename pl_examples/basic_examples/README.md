## Basic Examples

Use these examples to test how Lightning works.

#### MNIST

Trains MNIST where the model is defined inside the `LightningModule`.

```bash
# cpu
python simple_image_classifier.py

# gpus (any number)
python simple_image_classifier.py --trainer.gpus 2

# Distributed Data Parallel
python simple_image_classifier.py --trainer.gpus 2 --trainer.accelerator ddp
```

______________________________________________________________________

#### MNIST with DALI

The MNIST example above using [NVIDIA DALI](https://developer.nvidia.com/DALI).
Requires NVIDIA DALI to be installed based on your CUDA version, see [here](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).

```bash
python dali_image_classifier.py
```

______________________________________________________________________

#### Image classifier

Generic image classifier with an arbitrary backbone (ie: a simple system)

```bash
# cpu
python backbone_image_classifier.py

# gpus (any number)
python backbone_image_classifier.py --trainer.gpus 2

# Distributed Data Parallel
python backbone_image_classifier.py --trainer.gpus 2 --trainer.accelerator ddp
```

______________________________________________________________________

#### Autoencoder

Showing the power of a system... arbitrarily complex training loops

```bash
# cpu
python autoencoder.py

# gpus (any number)
python autoencoder.py --trainer.gpus 2

# Distributed Data Parallel
python autoencoder.py --trainer.gpus 2 --trainer.accelerator ddp
```
