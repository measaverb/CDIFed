import importlib
import inspect
import os
from argparse import Namespace

from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import (
    FederatedDataset,
)


def get_all_models():
    return [
        model.split(".")[0]
        for model in os.listdir(
            "fedml_api/standalone/domain_generalization/datasets"
        )
        if not model.find("__") > -1 and "py" in model
    ]


NAMES = {}
for model in get_all_models():
    mod = importlib.import_module(
        "fedml_api.standalone.domain_generalization.datasets." + model
    )
    dataset_classes_name = [
        x
        for x in mod.__dir__()
        if "type" in str(type(getattr(mod, x)))
        and "ContinualDataset" in str(inspect.getmro(getattr(mod, x))[1:])
    ]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c

    gcl_dataset_classes_name = [
        x
        for x in mod.__dir__()
        if "type" in str(type(getattr(mod, x)))
        and "GCLDataset" in str(inspect.getmro(getattr(mod, x))[1:])
    ]
    for d in gcl_dataset_classes_name:
        c = getattr(mod, d)
        NAMES[c.NAME] = c


def get_dataset(args: Namespace) -> FederatedDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
