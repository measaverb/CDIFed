import copy
import itertools
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# --- Assuming these imports are correctly set up in the user's environment ---
from fedml_api.standalone.domain_generalization.models.utils.clip_adapter import (
    CustomCLIP,
    create_clip_cfg,
    load_clip_to_cpu,
)
from fedml_api.standalone.domain_generalization.models.utils.federated_model import (
    FederatedModel,
)


# MODIFICATION: Renamed class from CDIFedResNet to CDIFedClassifier
class CDIFedClassifier(FederatedModel):
    # MODIFICATION: Updated NAME attribute
    NAME = "cdifedclassifier"

    def __init__(self, nets_list, args, transform):
        # nets_list is now expected to be a list of single-layer classifiers (e.g., nn.Linear)
        super(CDIFedClassifier, self).__init__(nets_list, args, transform)

        self.args.adapter_epochs = getattr(self.args, "adapter_epochs", 3)
        # MODIFICATION: Renamed distill_epochs to local_epochs for clarity, as there is no distillation
        self.args.local_epochs = getattr(
            self.args, "distill_epochs", self.args.local_epoch
        )
        self.args.lr_student = getattr(self.args, "lr_student", 1e-4)
        # MODIFICATION: Removed distill_lambda as it's no longer needed
        # self.args.distill_lambda = getattr(self.args, "distill_lambda", 1.0)

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
        # MODIFICATION: self.global_net is now the global classifier
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        self.base_clip_model = None
        self.tuned_adapters = {}
        # MODIFICATION: Projectors are client-specific and not aggregated
        self.projectors = nn.ModuleDict()

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()
        self.online_clients = online_clients

        for i in tqdm(online_clients, desc="Local Client Training", position=0):
            # MODIFICATION: self.nets_list[i] is the classifier for client i
            self._train_net(i, self.nets_list[i], priloader_list[i])
            # The saved checkpoint is for the local classifier
            self._save_checkpoint(
                self.nets_list[i].state_dict(),
                "local_clients",
                f"client_{i}_latest.pth",
            )

        self.aggregate_nets()
        print("Saving latest global model (classifier)...")
        self._save_checkpoint(
            self.global_net.state_dict(), "global", "latest_global.pth"
        )

        return None

    # MODIFICATION: Added explicit aggregation method for clarity
    def aggregate_nets(self):
        """
        Performs federated averaging on the classifiers of online clients.
        Projectors are not aggregated.
        """
        print("Aggregating classifier weights...")
        global_w = self.global_net.state_dict()
        
        # Zero out the global weights
        for key in global_w:
            global_w[key].zero_()

        online_classifiers = [self.nets_list[i] for i in self.online_clients]
        
        # Sum the weights of online client classifiers
        for classifier in online_classifiers:
            net_w = classifier.state_dict()
            for key in global_w:
                # Assuming equal weight for each client
                global_w[key] += net_w[key]
        
        # Average the weights
        for key in global_w:
            global_w[key] /= len(self.online_clients)

        # Load the new global weights into the global classifier
        self.global_net.load_state_dict(global_w)

        # Distribute the new global classifier to all clients for the next round
        for net in self.nets_list:
            net.load_state_dict(global_w)
        print("Aggregation complete. Global classifier distributed to all clients.")

    def _train_net(self, client_idx, classifier, train_loader):
        # =================================================================================
        # PHASE 1: ONE-TIME CLIP ADAPTER TUNING (Remains the same)
        # The adapter personalizes the feature extractor for each client's domain.
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
                    optimizer_adapter.step()
                    loop.set_postfix(loss=loss.item())
                print(f"[Client {client_idx}] Adapter training loss for epoch {epoch+1}: {avg_loss / len(loop)}")

            self.tuned_adapters[client_idx] = copy.deepcopy(
                tuning_model.adapter.state_dict()
            )
            print(f"[Client {client_idx}] Finished tuning. Stored adapter weights.")
            del tuning_model, optimizer_adapter
            torch.cuda.empty_cache()

        # =================================================================================
        # MODIFICATION: PHASE 2: PROJECTOR AND CLASSIFIER TRAINING
        # The adapter is frozen. We train a local projector and a classifier that will be aggregated.
        # =================================================================================
        print(f"[Client {client_idx}] Starting Phase 2: Projector and Classifier Training...")

        # Load the client's tuned adapter and freeze the entire teacher model
        client_adapter_state = self.tuned_adapters[client_idx]
        self.base_clip_model.adapter.load_state_dict(client_adapter_state)
        self.base_clip_model.eval()
        for param in self.base_clip_model.parameters():
            param.requires_grad = False

        classifier.train()

        str_client_idx = str(client_idx)
        if str_client_idx not in self.projectors:
            print(f"Initializing feature projector for client {client_idx}...")
            clip_feature_dim = self.base_clip_model.image_encoder_base.output_dim
            # The projector maps from CLIP's feature dimension to the classifier's input dimension
            classifier_in_features = classifier.in_features

            projector = nn.Sequential(
                nn.Linear(
                    clip_feature_dim, (clip_feature_dim + classifier_in_features) // 2
                ),
                nn.ReLU(),
                nn.Linear(
                    (clip_feature_dim + classifier_in_features) // 2, classifier_in_features
                ),
            ).to(self.device)
            self.projectors[str_client_idx] = projector

        projector = self.projectors[str_client_idx]
        projector.train()

        # Optimizer now trains only the projector and the classifier
        optimizer = Adam(
            itertools.chain(classifier.parameters(), projector.parameters()),
            lr=self.args.lr_student,
            weight_decay=0,
        )

        classification_loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.args.local_epochs):
            loop = tqdm(
                train_loader,
                desc=f"Classifier Epoch {epoch+1}/{self.args.local_epochs}",
                leave=False,
                position=1,
            )
            for batch in loop:
                image, label = batch[0].to(self.device), batch[1].to(self.device)

                optimizer.zero_grad()

                # 1. Get frozen features from the adapted CLIP model
                with torch.no_grad():
                    teacher_features = self.base_clip_model.encode_image(image)

                # 2. Pass features through the trainable local projector
                projected_features = projector(teacher_features.float())
                
                # 3. Get predictions from the trainable classifier
                outputs = classifier(projected_features)

                # 4. Calculate classification loss
                loss = classification_loss_fn(outputs, label)

                loss.backward()
                optimizer.step()

                loop.set_postfix(
                    cls_loss=loss.item()
                )