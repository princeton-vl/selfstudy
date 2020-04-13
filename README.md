# How Useful is Self-Supervised Pretraining for Visual Tasks?

Framework to reproduce experiments in:

> **How Useful is Self-Supervised Pretraining for Visual Tasks?** <br/>
>   [Alejandro Newell](https://www.alejandronewell.com/) and [Jia Deng](https://www.cs.princeton.edu/~jiadeng/). CVPR, 2020.
[arXiv:2003.14323](https://arxiv.org/abs/2003.14323)

## Setup

Ensure `selfstudy` is in your PYTHONPATH. Everything was tested with Python 3.7, all additional package and version information can be found in `requirements.txt`. More recent versions of python packages are likely to work, but have not been tested.

By default, it is assumed that experiments and data will be found in this directory, but if you wish to keep data and experiment results elsewhere adjust the relevant lines in `paths.py`.

All code was tested on an NVIDIA GeForce RTX 2080 Ti.

**Data:** Synthetic data generation code can be found [here](https://www.github.com/princeton-vl/selfstudy-render).
You can also download some pre-rendered datasets [here](https://drive.google.com/drive/folders/1veqmGOjB6-WE_WeXz0Sb7wHCq2WtbuMj?usp=sharing).

## Model Training

#### Starting with CIFAR

By default, the code runs on CIFAR out-of-the-box so you can test it without downloading or rendering the synthetic datasets.

Call `python main.py` and make sure nothing crashes. This runs a quick training session on CIFAR10. It is largely based off of the example provided [here](https://myrtle.ai/how-to-train-your-resnet), which provides a simple setup for training a small model on CIFAR as quickly as possible.

Attach the argument `-e, --exp_id` to provide a name for your experiment which determines where the output is placed in `paths.EXP_DIR`.

#### Gin configs

There are a lot of options that govern how this code will run, these are mostly managed by gin configs. Take a look at `config/cifar/r9/classify.gin` for an example. The config file controls the downstream task, dataset, data augmentation, optimization hyperparameters, etc. Change any options in config files as necessary to suit your experiment needs. To run code with a particular config, call: `python main.py -g [CONFIG_NAME]` (e.g. `python main.py -g cifar/r9/amdim`).

Individual parameters can be overridden via the command line with `-p`. Each parameter can be written out in the same format as in the config file in quotes and separated by spaces. For example:

`python main.py -p 'init_resnet.choice = "resnet50"' 'dataloader.batch_size = 128'`

Any function anywhere in the code can be made configurable, check out the [gin-config repo](https://github.com/google/gin-config) for more information.

#### Synthetic data

Config files are provided for pretraining models on the synthetic datasets. Those found in `config/syn` are for training on single-object datasets, while those in `config/dense` are for training on higher resolution multi-object images. Example pre-rendered datasets are available here. To generate datasets yourself, check out the [dataset rendering code](https://www.github.com/princeton-vl/selfstudy-render). All synthetic datasets are expected to be contained in `paths.DATA_DIR/syn/`.

The final post-processed output of the data rendering code is a single HDF5 file that contains the images and their corresponding labels. These images have already been normalized and resized and potentially converted to LAB color space. This is to alleviate the need to do any of this pre-processing during training.

Note, all data augmentation is performed on the GPU. This does take up a bit of GPU memory that might otherwise be devoted to a larger batchsize. With very large datasets it is impossible to fit the whole dataset at once on the GPU, so this is controlled with the configuration option `BaseDataset.max_images` which specifies how many images are processed at any given time on the GPU. If running out of GPU memory is an issue, you can either adjust this setting or `dataloader.batch_size`.

#### Ray configs

[Ray](https://github.com/ray-project/ray) is used for managing experiments. Check out `config/single_run.py` and `config/grid_search.py` for examples on the options that can be adjusted. The most important are namely those that pertain to saving checkpoints and performing searches over parameters. You can override gin config options in the ray config file and importantly, define sweeps over variables. It is also possible to define search spaces from which to do hyperparameter tuning as can be seen in the included examples.

Since ray configs can conveniently generate sets of hundreds of experiments, some helper functions are available in `util/vis.py` for loading and organizing final results from these experiments.

To run a version of an experiment with less overhead from Ray, you can call the code in local mode with the command line argument `-l, --local_mode`.

#### Model checkpoints

One thing about Ray is that snapshots are saved in subdirectories with long names and random hashes added to them. To load weights of a pretrained model, the code expects a path to be provided directly to the snapshot directory. As such, there is a little utility script that moves snapshots to more convenient directories.

```console
~/selfstudy$ python main.py -e cifar_test_0000
~/selfstudy$ python misc/move_snapshots.py -e cifar_test_0000 -m
Moving from: ~/selfstudy/exp/cifar_test_0000/e_0_2020-04-13_14-00-36ez5rvdao/checkpoint_25 to: ~/selfstudy/exp/cifar_test_0000/
```

If you want to sanity check that the move will be performed correctly, exclude `-m` to list the source and target directories but prevent the move command from actually being executed.

Now, let's say you have trained a dozen models across a sweep of datasets using the `grid_search` ray config. Individual parameters can be used to organize the snapshots with a formatting string. For example:

```console
~/selfstudy$ python misc/move_snapshots.py -e pretrain-rotate -p '{cfg[SynDataset.dataset_choice]}' -m
Moving from: ~/selfstudy/exp/pretrain-rotate/e_SynDataset.dataset_choice=xlg0010_0223_jh5ve3q9/checkpoint_400 to: ~/selfstudy/exp/pretrain-rotate/xlg0010
Moving from: ~/selfstudy/exp/pretrain-rotate/e_SynDataset.dataset_choice=xlg0011_0223_9fcu8rbg/checkpoint_400 to: ~/selfstudy/exp/pretrain-rotate/xlg0011
Moving from: ~/selfstudy/exp/pretrain-rotate/e_SynDataset.dataset_choice=xlg0100_0223_5_ugcw7b/checkpoint_400 to: ~/selfstudy/exp/pretrain-rotate/xlg0100
```

To then load one of those pretrained models during training, use:

`python main.py -p 'Wrapper.pretrained = "pretrain-rotate/xlg0010"'`

#### Test iterations

To run inference on the test set, set `session.test_iters = -1`.
