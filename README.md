# Mitigating Domain Shifts in Federated Learning using CLIP

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


## Training & Evaluation

To train the model(s) in the paper, run this command:

```train
python ./fedml_experiments/standalone/domain_generalization/main.py \
       --model cdifed
       --dataset fl_officecaltech
       --backbone resnet18
```


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