import importlib
import inspect
import os
from argparse import Namespace

from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import (
    FederatedDataset,
)
from fedml_api.standalone.domain_generalization.datasets.utils.public_dataset import (
    PublicDataset,
)


def get_all_models():
    return [
        model.split(".")[0]
        for model in os.listdir(
            "fedml_api/standalone/domain_generalization/datasets"
        )
        if not model.find("__") > -1 and "py" in model
    ]


Priv_NAMES = {}
for model in get_all_models():
    mod = importlib.import_module(
        "fedml_api.standalone.domain_generalization.datasets." + model
    )
    dataset_classes_name = [
        x
        for x in mod.__dir__()
        if "type" in str(type(getattr(mod, x)))
        and "FederatedDataset" in str(inspect.getmro(getattr(mod, x))[1:])
    ]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        Priv_NAMES[c.NAME] = c

Pub_NAMES = {}
for model in get_all_models():
    mod = importlib.import_module(
        "fedml_api.standalone.domain_generalization.datasets." + model
    )
    dataset_classes_name = [
        x
        for x in mod.__dir__()
        if "type" in str(type(getattr(mod, x)))
        and "PublicDataset" in str(inspect.getmro(getattr(mod, x))[1:])
    ]
    for d in dataset_classes_name:
        c = getattr(mod, d)
        Pub_NAMES[c.NAME] = c


def get_prive_dataset(args: Namespace) -> FederatedDataset:

    assert args.dataset in Priv_NAMES.keys()
    return Priv_NAMES[args.dataset](args)


def get_public_dataset(args: Namespace) -> PublicDataset:

    assert args.public_dataset in Pub_NAMES.keys()
    return Pub_NAMES[args.public_dataset](args)
