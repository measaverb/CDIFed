# CLIP-assisted Domain-invariant Representation Learning on Heterogeneous Federated learning

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

We use <a href="https://wandb.ai/">wandb</a> to keep a log of our experiments.
If you don't have a wandb account, just install it and use it as offline mode.

```wandb
pip install wandb
wandb off
```

[//]: # ">📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc..."

## Training & Evaluation

To train the model(s) in the paper, run this command:

```train
python ./fedml_experiments/standalone/domain_generalization/main.py \
       --model cdifed
       --dataset fl_officecaltech
       --backbone resnet18
```

[//]: # ">📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters."

## Arguments

You can modify the arguments to run CDIFed on other settings. The arguments are described as follows:

| Arguments             | Description                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `prefix`              | A prefix for logging.                                                                                             |
| `communication_epoch` | Total communication rounds of Federated Learning.                                                                 |
| `local_epoch`         | Local epochs for local model updating.                                                                            |
| `parti_num`           | Number of participants.                                                                                           |
| `model`               | Name of FL framework.                                                                                             |
| `dataset`             | Datasets used in the experiment. Options: `fl_officecaltech`, `fl_digits`.                                        |
| `backbone`            | Backbone global model. Options: `resnet10`, `resnet18`.                                                           |

[//]: # ">📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below)."
[//]: # "## Pre-trained Models"
[//]: #
[//]: # "You can download pretrained models here:"
[//]: #
[//]: # "- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. "
[//]: #
[//]: # ">📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models."
[//]: # "## Results"
[//]: #
[//]: # "Our model achieves the following performance on :"
[//]: #
[//]: # "### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)"
[//]: #
[//]: # "| Model name         | Top 1 Accuracy  | Top 5 Accuracy |"
[//]: # "| ------------------ |---------------- | -------------- |"
[//]: # "| My awesome model   |     85%         |      95%       |"
[//]: #
[//]: # ">📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. "
[//]: # "## Contributing"
[//]: #
[//]: # ">📋  Pick a licence and describe how to contribute to your code repository. "
