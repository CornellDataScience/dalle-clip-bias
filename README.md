# Bias testing in Text-to-Image models.

This repo is for running Text-to-Image models, bias testing them, releasing
web applications for demos, and doing general explorations of biases in ML.

Some example images:
<Insert image here>

## Set up
We use poetry + pyenv to set up the environment. However a requirements.txt was
made for those that don't use poetry + pyenv.

Create a new virtual Python environment for DALLE/CLIP:
```bash
# For pyenv + poetry
$ pyenv virtualenv 3.8.12 dalle-clip-bias
$ pyenv local dalle-clip-bias
$ poetry install
# For basic python
$ mkdir env
$ python -m venv env/
$ source env/bin/activate
$ pip install -r requirements.txt
```

There are required repositories to load before things can fully run. These vary
based on which method you are doing. There are separate instructions for each:

1. CLIP+Optim: Clone the required repositories
```bash
$ cd CLIP+Optim
$ git clone 'https://github.com/openai/CLIP'
```
2. CLIP+VQGAN: Clone the required repositores and make edits:
```bash
$ cd CLIP+Optim
$ git clone 'https://github.com/openai/CLIP'
$ git clone 'https://github.com/CompVis/taming-transformers' tamingtransformers
```

Then we need to modify files within `tamingtransformers/taming` to have their
imports changed. The following files need modifications:

* `tamingtransformers/taming/models/cond_transformer.py`
* `tamingtransformers/taming/models/vqgan.py`
* `tamingtransformers/taming/models/modules/discriminator/model.py`
* `tamingtransformers/taming/models/modules/discriminator/model.py`
* `tamingtransformers/taming/models/modules/losses/__init__.py`
* `tamingtransformers/taming/models/modules/discriminator/model.py`
* `tamingtransformers/taming/models/modules/losses/lpips.py`
* `tamingtransformers/taming/models/modules/losses/vqperceptual.py`

For each of these files, for any imports from `taming` or `main` they should be
changed to `tamingtransformers.taming` or `tamingtransformers.main`.

Lastly we will need to download some checkpoint files. E.g.

```bash
mkdir checkpoints

curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```

Then we must modify `checkpoints/vqgan_imagenet_f16_16384.yaml` to modify any
`taming` names into `tamingtransformers.taming`.

3. DALLE:
4. DALLE-mini:
