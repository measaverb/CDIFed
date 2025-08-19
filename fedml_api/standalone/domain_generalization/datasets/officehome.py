import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, ImageFolder

from fedml_api.standalone.domain_generalization.backbone.efficientnet import (
    EfficientNetB0,
)
from fedml_api.standalone.domain_generalization.backbone.googlenet import GoogLeNet
from fedml_api.standalone.domain_generalization.backbone.mobilnet_v2 import MobileNetV2
from fedml_api.standalone.domain_generalization.backbone.ResNet import (
    resnet10,
    resnet12,
    resnet18,
    resnet34,
    resnet50
)
from fedml_api.standalone.domain_generalization.datasets.transforms.denormalization import (
    DeNormalize,
)
from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import (
    FederatedDataset,
    partition_office_domain_skew_loaders_new,
)
from fedml_api.standalone.domain_generalization.utils.conf import data_path


class ImageFolder_Custom(DatasetFolder):
    def __init__(
        self,
        data_name,
        root,
        train=True,
        transform=None,
        target_transform=None,
        subset_train_num=7,
        subset_capacity=10,
    ):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(
                self.root + "OfficeHome/train/" + self.data_name + "/",
                self.transform,
                self.target_transform,
            )
        else:
            self.imagefolder_obj = ImageFolder(
                self.root + "OfficeHome/test/" + self.data_name + "/",
                self.transform,
                self.target_transform,
            )

        all_data = self.imagefolder_obj.samples
        self.train_index_list = []
        self.test_index_list = []
        self.classnames = [
            self.imagefolder_obj.classes[i]
            for i in range(len(self.imagefolder_obj.classes))
        ]
        for i in range(len(all_data)):
            if i % subset_capacity <= subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        if self.train:
            return len(self.train_index_list)
        else:
            return len(self.test_index_list)

    def __getitem__(self, index):

        if self.train:
            used_index_list = self.train_index_list
        else:
            used_index_list = self.test_index_list

        path = self.imagefolder_obj.samples[used_index_list[index]][0]
        target = self.imagefolder_obj.samples[used_index_list[index]][1]
        target = int(target)
        img = self.imagefolder_obj.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FedLeaOfficeCaltech(FederatedDataset):
    NAME = "fl_officehome"
    SETTING = "domain_skew"
    DOMAINS_LIST = ["Art", "Clipart", "Product", "Real_World"]
    percent_dict = {"Art": 0.2, "Clipart": 0.2, "Product": 0.2, "Real_World": 0.2}

    N_SAMPLES_PER_Class = None
    N_CLASS = 65
    Nor_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    def get_data_loaders(self, selected_domain_list):

        using_list = (
            self.DOMAINS_LIST if selected_domain_list == [] else selected_domain_list
        )

        nor_transform = self.Nor_TRANSFORM
        train_dataset_list = []
        test_dataset_list = []
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.get_normalization_transform(),
            ]
        )

        for _, domain in enumerate(using_list):
            train_dataset = ImageFolder_Custom(
                data_name=domain, root=data_path(), train=True, transform=nor_transform
            )

            train_dataset_list.append(train_dataset)

        for _, domain in enumerate(self.DOMAINS_LIST):
            test_dataset = ImageFolder_Custom(
                data_name=domain,
                root=data_path(),
                train=False,
                transform=test_transform,
            )
            test_dataset_list.append(test_dataset)

        traindls, testdls = partition_office_domain_skew_loaders_new(
            train_dataset_list, test_dataset_list, self
        )
        return traindls, testdls

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaOfficeCaltech.Nor_TRANSFORM]
        )
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list):
        nets_dict = {
            "res10": resnet10,
            "resnet10": resnet10,
            "resnet12": resnet12,
            "res18": resnet18,
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "efficient": EfficientNetB0,
            "mobilnet": MobileNetV2,
            "googlenet": GoogLeNet,
        }
        nets_list = []
        if names_list == None:
            for j in range(parti_num):
                # nets_list.append(resnet10(FedLeaOfficeCaltech.N_CLASS))
                nets_list.append(resnet18(FedLeaOfficeCaltech.N_CLASS))
        else:
            for j in range(parti_num):
                # net_name = names_list[j]
                net_name = names_list
                nets_list.append(nets_dict[net_name](FedLeaOfficeCaltech.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return transform
