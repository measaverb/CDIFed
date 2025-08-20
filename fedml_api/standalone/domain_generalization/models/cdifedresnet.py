import copy
import itertools
import os

import torch
import torch.nn as nn
import torchvision.models as models  # --- MODIFICATION ---: Import torchvision models
from torch.optim import Adam
from tqdm import tqdm

from fedml_api.standalone.domain_generalization.models.utils.federated_model import (
    FederatedModel,
)


class CDIFedResNet(FederatedModel):
    NAME = "cdifedresnet"

    def __init__(self, nets_list, args, transform):
        super(CDIFedResNet, self).__init__(nets_list, args, transform)

        # --- MODIFICATION ---: Removed adapter_epochs as there is no adapter tuning
        self.args.distill_epochs = getattr(
            self.args, "distill_epochs", self.args.local_epoch
        )
        self.args.lr_student = getattr(self.args, "lr_student", 1e-4)
        self.args.distill_lambda = getattr(self.args, "distill_lambda", 1.0)

        self.save_dir = getattr(self.args, "save_dir", "checkpoints")
        self.best_acc = 0.0
        if self.save_dir:
            os.makedirs(os.path.join(self.save_dir, "global"), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "local_clients"), exist_ok=True)
            print(f"Model checkpoints will be saved in '{self.save_dir}'")

    def _save_checkpoint(self, state_dict, folder, filename):
        if not self.save_dir:
            return
        filepath = os.path.join(self.save_dir, folder, filename)
        torch.save(state_dict, filepath)
        print(f"Saved checkpoint: {filepath}")

    def save_best_global_model_if_better(self, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            print(f"New best global model found. Accuracy: {acc:.4f}. Saving...")
            self._save_checkpoint(
                self.global_net.state_dict(), "global", "best_global.pth"
            )

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        # --- MODIFICATION ---: Replaced CLIP with a standard teacher model
        self.teacher_model = None
        self.projectors = nn.ModuleDict()

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        for i in tqdm(online_clients, desc="Local Client Training", position=0):
            self._train_net(i, self.nets_list[i], priloader_list[i])
            self._save_checkpoint(
                self.nets_list[i].state_dict(),
                "local_clients",
                f"client_{i}_latest.pth",
            )

        self.aggregate_nets(None)
        print("Saving latest global model...")
        self._save_checkpoint(
            self.global_net.state_dict(), "global", "latest_global.pth"
        )

        return None

    def _train_net(self, client_idx, net, train_loader):
        print(f"\n[Client {client_idx}] Starting training process...")

        # =================================================================================
        # --- MODIFICATION ---: ONE-TIME TEACHER MODEL INITIALIZATION
        # Replaced the entire CLIP adapter tuning phase with a simple, one-time
        # initialization of a frozen ResNet-50 teacher model.
        # =================================================================================
        if self.teacher_model is None:
            print("Initializing shared ImageNet-pretrained ResNet-50 (teacher)...")
            # Load a pretrained ResNet-50
            teacher = models.resnet50(pretrained=True)
            # Create a feature extractor by removing the final classification layer (fc)
            self.teacher_model = nn.Sequential(*list(teacher.children())[:-1])
            self.teacher_model.to(self.device)

            # Freeze the teacher model as we don't want to train it
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            print("Teacher model initialized, frozen, and set to evaluation mode.")

        # =================================================================================
        # PHASE 2: RESNET-50 FEATURE DISTILLATION TO RESNET-18 (Student)
        # =================================================================================
        print(f"[Client {client_idx}] Starting Phase 2: Feature Distillation...")

        # Set the student model to training mode
        net.train()

        str_client_idx = str(client_idx)
        if str_client_idx not in self.projectors:
            print(f"Initializing feature projector for client {client_idx}...")
            # --- MODIFICATION ---: Updated feature dimensions for ResNet-50 and ResNet-18
            teacher_feature_dim = 2048  # ResNet-50 feature output dimension
            student_feature_dim = net.cls.in_features  # ResNet-18 is 512

            projector = nn.Sequential(
                nn.Linear(
                    teacher_feature_dim,
                    (teacher_feature_dim + student_feature_dim) // 2,
                ),
                nn.ReLU(),
                nn.Linear(
                    (teacher_feature_dim + student_feature_dim) // 2,
                    student_feature_dim,
                ),
            ).to(self.device)
            self.projectors[str_client_idx] = projector

        projector = self.projectors[str_client_idx]
        projector.train() # The projector must be trained alongside the student

        # Optimizer for both the student network (ResNet-18) and the projector
        optimizer = Adam(
            itertools.chain(net.parameters(), projector.parameters()),
            lr=self.args.lr_student,
            weight_decay=0,
        )

        classification_loss_fn = nn.CrossEntropyLoss()
        distillation_loss_fn = nn.MSELoss()

        for epoch in range(self.args.distill_epochs):
            loop = tqdm(
                train_loader,
                desc=f"Distill Epoch {epoch+1}/{self.args.distill_epochs}",
                leave=False,
                position=1,
            )
            for batch in loop:
                image, label = batch[0].to(self.device), batch[1].to(self.device)

                optimizer.zero_grad()

                # Get features and classification outputs from the student (ResNet-18)
                student_features = net.features(image)
                outputs = net.cls(student_features)

                # Get features from the frozen teacher (ResNet-50)
                with torch.no_grad():
                    teacher_features_raw = self.teacher_model(image)
                    # Flatten the output of the pooling layer (e.g., from [N, 2048, 1, 1] to [N, 2048])
                    teacher_features = torch.flatten(teacher_features_raw, 1)

                # Project teacher features to match student feature dimension
                projected_teacher_features = projector(teacher_features.float())

                # Calculate losses
                class_loss = classification_loss_fn(outputs, label)
                distill_loss = distillation_loss_fn(
                    student_features, projected_teacher_features
                )

                # Combine losses
                total_loss = class_loss + self.args.distill_lambda * distill_loss

                total_loss.backward()
                optimizer.step()

                loop.set_postfix(
                    cls_loss=class_loss.item(), dist_loss=distill_loss.item()
                )