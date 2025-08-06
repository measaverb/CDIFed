import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from fedml_api.standalone.domain_generalization.models.utils.cocoop import (
    CoCoOpTuner,
    CustomCLIP,
    load_clip_to_cpu,
)
from fedml_api.standalone.domain_generalization.models.utils.federated_model import (
    FederatedModel,
)


class CDIFed(FederatedModel):
    NAME = "cdifed"

    def __init__(self, nets_list, args, transform):
        super(CDIFed, self).__init__(nets_list, args, transform)

    def ini(self):
        self.global_net = nn.ModuleList([copy.deepcopy(net) for net in self.nets_list])
        global_w = [net.state_dict() for net in self.nets_list]
        for net in self.nets_list:
            net.load_state_dict(global_w[0])

        self.base_clip_model = None

        self.tuned_prompt_learners = {}

        self.cocoop_tuner = CoCoOpTuner(self.args, self.device)

        # TODO: Implement a mechanism to save/load self.tuned_prompt_learners

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        for i in tqdm(online_clients, desc="Local Updates"):
            self._train_net(i, self.nets_list[i], priloader_list[i])

        # Aggregation
        self.aggregate_nets(None)
        return None

    def _train_net(self, client_idx, net, train_loader):
        # --- Phase 1: One-Time CLIP Prompt Tuning on the local dataset ---
        print(f"\nStarting training for client {client_idx}...")
        classnames = train_loader.dataset.classnames
        print(classnames)

        if self.base_clip_model is None:
            print("Initializing shared base CLIP model...")
            base_clip = load_clip_to_cpu(self.cocoop_tuner.cfg)
            self.base_clip_model = CustomCLIP(
                self.cocoop_tuner.cfg, classnames, base_clip
            )
            self.base_clip_model.to(self.device)
            self.base_clip_model.eval()

        if client_idx not in self.tuned_prompt_learners:
            print(
                f"\nPerforming one-time CLIP prompt tuning for client {client_idx}..."
            )
            prompt_learner_state = self.cocoop_tuner.tune(train_loader, classnames)
            self.tuned_prompt_learners[client_idx] = prompt_learner_state
            print(f"Finished tuning for client {client_idx}. Stored prompt learner.")

        # --- Phase 2: CLIP Distillation on the local client model ---
        client_prompt_state = self.tuned_prompt_learners[client_idx]
        self.base_clip_model.prompt_learner.load_state_dict(client_prompt_state)

        print(f"\nProceeding to Phase 2 (Distillation) for client {client_idx}...")
        with torch.no_grad():
            pass
        pass
