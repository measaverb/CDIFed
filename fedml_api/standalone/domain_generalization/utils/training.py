import copy
from argparse import Namespace
from collections import Counter
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

import wandb
from fedml_api.standalone.domain_generalization.datasets.utils.federated_dataset import (
    FederatedDataset,
)
from fedml_api.standalone.domain_generalization.models.utils.federated_model import (
    FederatedModel,
)
from fedml_api.standalone.domain_generalization.utils.logger import CsvWriter


def global_evaluate(
    model: FederatedModel, test_dl: DataLoader, setting: str, name: str
) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                if model.NAME == "nefl":
                    outputs, _ = net(images)
                else:
                    outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        accs.append(top1acc)
    net.train(status)
    return accs


def visualise_tsne_features(
    model,
    test_dls: list,
    device: torch.device,
    title: str = "t-SNE Visualisation of Features by Domain",
):
    """
    Extracts features from a model for different domains, performs t-SNE using
    RAPIDS cuML, and visualizes the result.

    Args:
        model (FederatedModel): The federated model containing the global_net.
                                The global_net must have a `.features()` method.
        test_dls (list): A list of PyTorch DataLoaders, each for a different domain.
        device (torch.device): The device to run the model on (e.g., 'cuda:0').
        title (str): The title for the plot.
    """
    try:
        import cupy
        from cuml.manifold import TSNE
    except ImportError:
        print(
            "RAPIDS cuML and CuPy are not installed. Please install them to use this function."
        )
        print("See: https://rapids.ai/start.html")
        return

    print("Starting feature extraction...")
    all_features = []
    all_labels = []
    all_domains = []

    net = model.global_net
    status = net.training
    net.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for domain_id, dl in enumerate(test_dls):
            print(f"  - Processing Domain {domain_id+1}/{len(test_dls)}")
            for images, labels in dl:
                images = images.to(device)
                # Use the .features() method to get the feature representation
                features = net.features(images)

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                # Create an array of domain IDs for this batch
                domain_ids = np.full(labels.shape[0], domain_id)
                all_domains.append(domain_ids)

    net.train(status)  # Restore the model's original training status

    # Concatenate all batch results into single numpy arrays
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.concatenate(all_domains, axis=0)

    print(f"\nFeature extraction complete. Total samples: {all_features.shape[0]}")
    print("Running t-SNE on GPU with cuML...")

    features_gpu = cupy.asarray(all_features)

    tsne = TSNE(n_components=2, perplexity=50, n_iter=10000, random_state=42)
    tsne_results_gpu = tsne.fit_transform(features_gpu)

    tsne_results = tsne_results_gpu.get()

    print("t-SNE complete. Plotting results...")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 12))

    unique_domains = np.unique(all_domains)
    unique_classes = np.unique(all_labels)
    num_domains = len(unique_domains)

    # Define palettes and markers with swapped logic
    class_palette = sns.color_palette("tab10", n_colors=len(unique_classes))
    domain_markers = ["o", "s", "X", "^", "P", "D", "*", "v", "<", ">"]

    if num_domains > len(domain_markers):
        print(
            f"Warning: Number of domains ({num_domains}) is greater than the number of unique markers "
            f"({len(domain_markers)}). Markers will be reused."
        )

    # Plot each domain-class combination
    for domain_idx, domain_id in enumerate(unique_domains):
        for class_idx, class_id in enumerate(unique_classes):
            mask = (all_domains == domain_id) & (all_labels == class_id)
            if not np.any(mask):
                continue

            # Assign color by class and marker by domain
            color = class_palette[class_idx % len(class_palette)]
            marker = domain_markers[domain_idx % len(domain_markers)]

            ax.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                color=color,
                marker=marker,
                alpha=0.8,
                s=80,
                edgecolor="k",
                linewidth=0.5,
            )

    # --- Create Custom Legends (with swapped logic) ---
    # 1. Legend for Classes (Colors)
    # class_handles = [
    #     Line2D(
    #         [0],
    #         [0],
    #         marker="o",
    #         color="w",
    #         label=f"Class {c}",
    #         markerfacecolor=color,
    #         markersize=10,
    #     )
    #     for c, color in zip(unique_classes, class_palette)
    # ]
    # legend1 = ax.legend(
    #     handles=class_handles,
    #     title="Classes",
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 1),
    # )
    # ax.add_artist(legend1)

    # 2. Legend for Domains (Markers)
    # domain_handles = [
    #     Line2D(
    #         [0],
    #         [0],
    #         marker=domain_markers[i % len(domain_markers)],
    #         color="gray",
    #         linestyle="None",
    #         label=f"Domain {d}",
    #         markersize=10,
    #     )
    #     for i, d in enumerate(unique_domains)
    # ]
    # ax.legend(
    #     handles=domain_handles,
    #     title="Domains",
    #     loc="lower left",
    #     bbox_to_anchor=(1.02, 0),
    # )

    # ax.set_title(title, fontsize=20)
    # ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
    # ax.set_ylabel("t-SNE Dimension 2", fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("tsne_visualisation.png")


def local_evaluate(
    model: FederatedModel,
    test_dl: DataLoader,
    domains_list: list,
    selected_domain_list: list,
    setting: str,
    name: str,
) -> list:
    all_accs = {}
    for i, net in enumerate(model.nets_list):
        status = net.training
        net.eval()
        domain = selected_domain_list[i]
        domain_index = domains_list.index(domain)
        if domain_index not in all_accs.keys():
            all_accs[domain_index] = []
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(test_dl[domain_index]):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        all_accs[domain_index].append(top1acc)
        net.train(status)

    avg_accs = []
    for i in range(len(all_accs)):
        avg_acc = round(sum(all_accs[i]) / len(all_accs[i]), 2)
        avg_accs.append(avg_acc)
    return avg_accs


def train(
    model: FederatedModel, private_dataset: FederatedDataset, args: Namespace
) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == "fl_officecaltech":
                selected_domain_list = np.random.choice(
                    domains_list,
                    size=args.parti_num - domains_len,
                    replace=True,
                    p=None,
                )
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == "fl_digits":
                # selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
                selected_domain_list = np.random.choice(
                    domains_list,
                    size=args.parti_num - domains_len,
                    replace=True,
                    p=None,
                )
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == "fl_domainnet":
                selected_domain_list = np.random.choice(
                    domains_list,
                    size=args.parti_num - domains_len,
                    replace=True,
                    p=None,
                )
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == "fl_officehome":
                selected_domain_list = np.random.choice(
                    domains_list,
                    size=args.parti_num - domains_len,
                    replace=True,
                    p=None,
                )
                selected_domain_list = list(selected_domain_list) + domains_list

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        selected_domain_dict = {"mnist": 6, "usps": 4, "svhn": 3, "syn": 7}

        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)
    print(result)

    print(selected_domain_list)
    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(
        selected_domain_list
    )
    model.trainloaders = pri_train_loaders
    if hasattr(model, "ini"):
        model.ini()

    accs_dict = {}
    mean_accs_list = []
    best_acc = 0
    best_accs = []

    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index

        if hasattr(model, "loc_update"):
            epoch_loc_loss_dict = model.loc_update(pri_train_loaders)

        if args.model in ["localtest"]:
            accs = local_evaluate(
                model,
                test_loaders,
                domains_list,
                selected_domain_list,
                private_dataset.SETTING,
                private_dataset.NAME,
            )
            model.aggregate_nets()
        else:
            accs = global_evaluate(
                model, test_loaders, private_dataset.SETTING, private_dataset.NAME
            )

        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        if mean_acc > best_acc:
            best_acc = mean_acc
        if len(best_accs) == 0:
            best_accs = copy.deepcopy(accs)
        for i in range(len(accs)):
            if accs[i] > best_accs[i]:
                best_accs[i] = accs[i]

        if args.wandb:
            wandb.log(
                {"Best_Acc": best_acc, "Mean_Acc": mean_acc, "round": epoch_index}
            )
            if len(best_accs) == 0:
                best_accs = copy.deepcopy(accs)
            for i in range(len(accs)):
                name = "Domain" + str(i)
                wandb.log(
                    {
                        name + "_Acc": accs[i],
                        name + "_BestAcc": best_accs[i],
                        "round": epoch_index,
                    }
                )

        print(
            "Round:",
            str(epoch_index),
            "Method:",
            model.args.model,
            "Mean_Acc:",
            str(mean_acc),
            "Best_Acc:",
            str(best_acc),
        )
        print("Domain_Acc:", accs, "Domain_BestAcc:", best_accs)

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)
