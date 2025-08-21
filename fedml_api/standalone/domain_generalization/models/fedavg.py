import copy

import torch.nn as nn
import torch.nn.utils.prune as torch_prune
import torch.optim as optim
from thop import profile
from torchstat import stat
from tqdm import tqdm

from fedml_api.standalone.domain_generalization.models.utils.federated_model import (
    FederatedModel,
)


class FedAvg(FederatedModel):
    NAME = "fedavg"

    def __init__(self, nets_list, args, transform):
        super(FedAvg, self).__init__(nets_list, args, transform)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])

        self.aggregate_nets(None)

        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(
            net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5
        )
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for i in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = net.features(images)
                outputs = net.classifier(features)

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
