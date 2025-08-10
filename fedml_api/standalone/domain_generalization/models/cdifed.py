import copy
import itertools

import torch
import torch.nn as nn
from torch.optim import Adam
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
        """
        Initializes the federated model, global network, and CoCoOp/distillation components.
        """
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        # --- CoCoOp Components ---
        self.base_clip_model = None
        self.tuned_prompt_learners = {}
        self.cocoop_tuner = CoCoOpTuner(self.args, self.device)

        # --- Distillation Components ---
        # Projectors will be created lazily, since we need model dimensions first
        self.projectors = (
            nn.ModuleDict()
        )  # Using ModuleDict to hold client-specific projectors

        # TODO: Implement a mechanism to save/load self.tuned_prompt_learners and self.projectors

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
        # =================================================================================
        # PHASE 1: ONE-TIME CLIP PROMPT TUNING (as before)
        # =================================================================================
        print(f"\nStarting training for client {client_idx}...")
        try:
            classnames = train_loader.dataset.classnames
        except AttributeError:
            classnames = train_loader.dataset.classes

        if self.base_clip_model is None:
            print("Initializing shared base CLIP model...")
            base_clip = load_clip_to_cpu(self.cocoop_tuner.cfg)
            self.base_clip_model = CustomCLIP(
                self.cocoop_tuner.cfg, classnames, base_clip
            )
            self.base_clip_model.to(self.device)
            self.base_clip_model.eval()  # Keep teacher frozen

        if client_idx not in self.tuned_prompt_learners:
            print(f"Performing one-time CLIP prompt tuning for client {client_idx}...")
            prompt_learner_state = self.cocoop_tuner.tune(train_loader, classnames)
            self.tuned_prompt_learners[client_idx] = prompt_learner_state
            print(f"Finished tuning for client {client_idx}. Stored prompt learner.")

        # =================================================================================
        # PHASE 2: CLIP FEATURE DISTILLATION
        # =================================================================================

        # Set the client's tuned prompt for the teacher model
        client_prompt_state = self.tuned_prompt_learners[client_idx]
        self.base_clip_model.prompt_learner.load_state_dict(client_prompt_state)

        net.train()  # Set the student model to training mode

        # --- Lazy Initialization of the Projector ---
        # Create a projector for this client if it doesn't exist
        str_client_idx = str(client_idx)
        if str_client_idx not in self.projectors:
            print(f"Initializing feature projector for client {client_idx}...")
            # Get dimensions
            clip_feature_dim = self.base_clip_model.image_encoder.output_dim
            # NOTE: Assumes your ResNet-18 has a '.fc' layer to get the feature dimension.
            # If your model structure is different, you must change this line.
            student_feature_dim = net.cls.in_features

            # Create the MLP and add it to our ModuleDict
            projector = nn.Sequential(
                nn.Linear(
                    clip_feature_dim, (clip_feature_dim + student_feature_dim) // 2
                ),
                nn.ReLU(),
                nn.Linear(
                    (clip_feature_dim + student_feature_dim) // 2, student_feature_dim
                ),
            ).to(self.device)
            self.projectors[str_client_idx] = projector

        # --- Optimizer for Distillation Phase ---
        # The optimizer needs to update BOTH the student network and its specific projector
        projector = self.projectors[str_client_idx]
        optimizer = Adam(
            itertools.chain(net.parameters(), projector.parameters()),
            lr=0.0001,
            weight_decay=0,
        )

        # --- Loss Functions ---
        classification_loss_fn = nn.CrossEntropyLoss()
        distillation_loss_fn = nn.MSELoss()

        print(
            f"Proceeding to Phase 2 (Feature Distillation) for client {client_idx}..."
        )
        for epoch in range(
            self.args.local_epoch
        ):  # Use general training epochs for distillation
            loop = tqdm(
                train_loader, desc=f"Distillation Epoch {epoch+1}/{self.args.local_epoch}"
            )
            for batch in loop:
                image, label = batch[0].to(self.device), batch[1].to(
                    self.device
                )

                optimizer.zero_grad()

                # --- Get Features ---
                # 1. Get student features from ResNet-18 (before the final classifier)
                # NOTE: This requires your ResNet model to have a method like `extract_features`.
                # If not, you must implement one. See notes below.
                student_features = net.features(image)

                # 2. Get teacher features from CLIP (with no gradients)
                with torch.no_grad():
                    teacher_features = self.base_clip_model.image_encoder(
                        image.type(self.base_clip_model.dtype)
                    )

                # 3. Project teacher features to match student's dimension
                projected_teacher_features = projector(teacher_features.float())

                # --- Calculate Losses ---
                # 1. Classification loss for the student model
                outputs = net.classifier(student_features)
                class_loss = classification_loss_fn(outputs, label)

                # 2. Distillation loss (MSE between student and projected teacher features)
                distill_loss = distillation_loss_fn(
                    student_features, projected_teacher_features
                )

                # 3. Combine losses with a weighting factor
                total_loss = class_loss + 1.0 * distill_loss

                # --- Backpropagation ---
                total_loss.backward()
                optimizer.step()

                loop.set_postfix(
                    class_loss=class_loss.item(), distill_loss=distill_loss.item()
                )
