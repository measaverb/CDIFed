import copy
import itertools
import os

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


class CDIFedClassifier(FederatedModel):
    NAME = "cdifedclassifier"

    def __init__(self, nets_list, args, transform):
        super(CDIFedClassifier, self).__init__(nets_list, args, transform)

        self.args.adapter_epochs = getattr(self.args, "adapter_epochs", 3)
        # <<< START OF MODIFICATION: RENAME EPOCH ARGUMENT FOR CLARITY >>>
        # The second phase is now for classifier training, not distillation.
        self.args.classifier_epochs = getattr(
            self.args, "distill_epochs", self.args.local_epoch
        )
        # <<< END OF MODIFICATION >>>
        self.args.lr_student = getattr(self.args, "lr_student", 1e-4)
        # distill_lambda is no longer used, but kept for config compatibility
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

    # <<< START OF MODIFICATION: ADDED AGGREGATION METHOD FOR CLASSIFIERS >>>
    def _aggregate_classifiers(self):
        """
        Aggregates the classifier weights from online clients and distributes
        the new global classifier back to all clients.
        """
        print("\nAggregating client classifiers...")
        # Assuming self.global_net.cls and client_net.cls exist and are the classifiers
        global_cls_dict = self.global_net.cls.state_dict()

        # Zero out the global classifier's state dict for accumulation
        for k in global_cls_dict.keys():
            global_cls_dict[k] = torch.zeros_like(global_cls_dict[k])

        # Sum the state dicts of all online clients' classifiers
        for i in self.online_clients:
            client_cls_dict = self.nets_list[i].cls.state_dict()
            for k in global_cls_dict.keys():
                global_cls_dict[k] += client_cls_dict[k]

        # Average the accumulated state dicts
        for k in global_cls_dict.keys():
            global_cls_dict[k] = torch.div(global_cls_dict[k], len(self.online_clients))

        # Load the averaged classifier into the global model
        self.global_net.cls.load_state_dict(global_cls_dict)
        print("Global classifier has been updated with the average of online clients.")

        # Distribute the new global classifier to all clients for the next round
        for net in self.nets_list:
            net.cls.load_state_dict(self.global_net.cls.state_dict())
        print("Distributed the new global classifier to all local clients.")
    # <<< END OF MODIFICATION >>>

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

        # <<< START OF MODIFICATION: REPLACE AGGREGATION LOGIC >>>
        # self.aggregate_nets(None) # Original: aggregates the entire network
        self._aggregate_classifiers()  # New: aggregates the classifier only
        # <<< END OF MODIFICATION >>>

        print("Saving latest global model...")
        self._save_checkpoint(
            self.global_net.state_dict(), "global", "latest_global.pth"
        )

        return None

    def _train_net(self, client_idx, net, train_loader):
        # =================================================================================
        # PHASE 1: ONE-TIME CLIP ADAPTER TUNING (UNCHANGED)
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
                total_loss, total_count = 0.0, 0
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
                    optimizer_adapter.step()

                    total_loss += loss.item() * image.size(0)
                    total_count += image.size(0)
                    loop.set_postfix(loss=loss.item())
                avg_loss = total_loss / total_count if total_count > 0 else 0
                print(f"[Client {client_idx}] Adapter training loss: {avg_loss:.4f}")

            self.tuned_adapters[client_idx] = copy.deepcopy(
                tuning_model.adapter.state_dict()
            )
            print(f"[Client {client_idx}] Finished tuning. Stored adapter weights.")
            del tuning_model, optimizer_adapter
            torch.cuda.empty_cache()

        # <<< START OF MODIFICATION: PHASE 2 IS NOW TRAINING THE CLASSIFIER >>>
        # =================================================================================
        # PHASE 2: TRAIN PROJECTOR AND CLASSIFIER
        # =================================================================================

        client_adapter_state = self.tuned_adapters[client_idx]
        self.base_clip_model.adapter.load_state_dict(client_adapter_state)
        self.base_clip_model.eval() # Teacher model (with adapter) is frozen

        print(f"[Client {client_idx}] Starting Phase 2: Projector and Classifier Training...")

        # Freeze the student's backbone and ensure only the classifier is trainable
        net.train()
        for param in net.parameters():
            param.requires_grad = False
        for param in net.cls.parameters():
            param.requires_grad = True

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
        projector.train()

        # Optimizer now only trains the projector and the classifier part of the student net
        optimizer = Adam(
            itertools.chain(net.cls.parameters(), projector.parameters()),
            lr=self.args.lr_student,
            weight_decay=0,
        )

        classification_loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.args.classifier_epochs):
            loop = tqdm(
                train_loader,
                desc=f"Classifier Epoch {epoch+1}/{self.args.classifier_epochs}",
                leave=False,
                position=1,
            )
            for batch in loop:
                image, label = batch[0].to(self.device), batch[1].to(self.device)

                optimizer.zero_grad()

                # 1. Get teacher features from the fine-tuned CLIP model
                with torch.no_grad():
                    teacher_features = self.base_clip_model.encode_image(image)

                # 2. Project teacher features into the student's feature space
                projected_teacher_features = projector(teacher_features.float())

                # 3. Pass projected features through the student's classifier method
                outputs = net.classifier(projected_teacher_features)

                # 4. Calculate classification loss
                class_loss = classification_loss_fn(outputs, label)

                # The total loss is just the classification loss
                total_loss = class_loss

                total_loss.backward()
                optimizer.step()

                loop.set_postfix(cls_loss=class_loss.item())
        # <<< END OF MODIFICATION >>>
