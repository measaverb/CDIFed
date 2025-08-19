import copy
import itertools
import os  # --- SAVING LOGIC ---: Import os for path manipulation

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from fedml_api.standalone.domain_generalization.models.utils.clip_adapter import (
    CustomCLIP,
    create_clip_cfg,
    load_clip_to_cpu,
)
from fedml_api.standalone.domain_generalization.models.utils.federated_model import (
    FederatedModel,
)


class CDIFed(FederatedModel):
    NAME = "cdifed"

    def __init__(self, nets_list, args, transform):
        super(CDIFed, self).__init__(nets_list, args, transform)

        self.args.adapter_epochs = getattr(self.args, "adapter_epochs", 3)
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

        self.base_clip_model = None
        self.tuned_adapters = {}
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
        # =================================================================================
        # PHASE 1: ONE-TIME CLIP ADAPTER TUNING
        # =================================================================================
        print(f"\n[Client {client_idx}] Starting training process...")
        try:
            classnames = train_loader.dataset.classnames
        except AttributeError:
            classnames = train_loader.dataset.classes

        if self.base_clip_model is None:
            print("Initializing shared base CLIP model (teacher)...")
            clip_cfg = create_clip_cfg(args=self.args)
            base_clip = load_clip_to_cpu(clip_cfg)
            base_clip.float()
            self.base_clip_model = CustomCLIP(classnames, base_clip)
            self.base_clip_model.to(self.device)
            self.base_clip_model.eval()

        if client_idx not in self.tuned_adapters:
            print(f"[Client {client_idx}] Performing one-time CLIP-Adapter tuning...")

            tuning_model = copy.deepcopy(self.base_clip_model)
            tuning_model.to(self.device)

            for name, param in tuning_model.named_parameters():
                param.requires_grad_("adapter" in name)

            tuning_model.adapter.train()

            optimizer_adapter = Adam(
                tuning_model.adapter.parameters(), lr=self.args.lr_clip
            )
            loss_fn_adapter = nn.CrossEntropyLoss()
            for epoch in range(self.args.adapter_epochs):
                avg_loss = 0
                loop = tqdm(
                    train_loader,
                    desc=f"Adapter Epoch {epoch+1}/{self.args.adapter_epochs}",
                    postfix={"adapter_loss": 0.0},
                    leave=False,
                    position=1,
                )
                for batch in loop:
                    image, label = batch[0].to(self.device), batch[1].to(self.device)

                    optimizer_adapter.zero_grad()
                    logits = tuning_model(image)
                    loss = loss_fn_adapter(logits, label)
                    loss.backward()
                    avg_loss += loss.item()
                    avg_loss /= batch[0].size(0)
                    optimizer_adapter.step()
                    loop.set_postfix(loss=loss.item())
                print(f"[Client {client_idx}] Adapter training loss: {avg_loss}")

            self.tuned_adapters[client_idx] = copy.deepcopy(
                tuning_model.adapter.state_dict()
            )
            print(f"[Client {client_idx}] Finished tuning. Stored adapter weights.")
            del tuning_model, optimizer_adapter
            torch.cuda.empty_cache()

        # =================================================================================
        # PHASE 2: CLIP FEATURE DISTILLATION (using the tuned adapter)
        # =================================================================================
        print(f"[Client {client_idx}] Starting Phase 2: Feature Distillation...")

        client_adapter_state = self.tuned_adapters[client_idx]
        self.base_clip_model.adapter.load_state_dict(client_adapter_state)
        self.base_clip_model.eval()

        net.train()

        str_client_idx = str(client_idx)
        if str_client_idx not in self.projectors:
            print(f"Initializing feature projector for client {client_idx}...")
            clip_feature_dim = self.base_clip_model.image_encoder_base.output_dim
            student_feature_dim = net.cls.in_features

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

        projector = self.projectors[str_client_idx]

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

                student_features = net.features(image)
                outputs = net.cls(student_features)

                with torch.no_grad():
                    teacher_features = self.base_clip_model.encode_image(image)

                projected_teacher_features = projector(teacher_features.float())

                class_loss = classification_loss_fn(outputs, label)
                distill_loss = distillation_loss_fn(
                    student_features, projected_teacher_features
                )

                total_loss = class_loss + self.args.distill_lambda * distill_loss

                total_loss.backward()
                optimizer.step()

                loop.set_postfix(
                    cls_loss=class_loss.item(), dist_loss=distill_loss.item()
                )
