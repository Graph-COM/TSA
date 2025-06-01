import csv
import json
import logging
import pdb
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor


class Metrics:
    def __init__(self, root, data_name):
        self.root = root
        self.metrics = defaultdict(list)
        self.data_name = data_name

    def calculate_metrics(self, probs: Tensor, labels: Tensor, name: str):

        # _, predicted = torch.max(probs, 1)

        probs = probs.cpu().detach().numpy()
        predicted = np.argmax(probs, axis=1)
        labels = labels.cpu().detach().numpy()

        if len(probs.shape) == 1 or probs.shape[1] == 2:
            probs_for_auc = probs[:, 1] if probs.shape[1] == 2 else probs
        else:
            probs_for_auc = probs
        # pdb.set_trace()
        if self.data_name == "MAG":
            idx = labels != len(np.unique(labels)) - 1
            self.metrics[name.lower() + "_accuracy"].append(
                accuracy_score(labels[idx], predicted[idx])
            )
        elif self.data_name == "Pileup":
            idx = (labels != 2) & (labels != 3)
            self.metrics[name.lower() + "_accuracy"].append(
                accuracy_score(labels[idx], predicted[idx])
            )
        else:
            self.metrics[name.lower() + "_accuracy"].append(
                accuracy_score(labels, predicted)
            )                        

        sample_weight = np.ones_like(labels, dtype=float)
        # sample_weight[labels == 19] = 0
        if self.data_name == "Pileup":
            idx = (labels != 2) & (labels != 3)
            b_label = labels[idx]
            b_predicted = predicted[idx]
            self.metrics[name.lower() + "_bal_accuracy"].append(
                balanced_accuracy_score(b_label, b_predicted)
            )
            # In order to calculate f1 scores in binary setting, we need to ensure
            # the predictions are 0 or 1.
            condition = ((b_predicted == 2) | (b_predicted == 3)) & (b_label == 0)
            b_predicted[condition] = 1
            condition = ((b_predicted == 2) | (b_predicted == 3)) & (b_label == 1)
            b_predicted[condition] = 0
            self.metrics[name.lower() + "_f1_score"].append(
                f1_score(
                    b_label,
                    b_predicted,
                    average="binary",
                )
            )
            self.metrics[name.lower() + "_rank_accuracy"].append(rank_accuracy(labels[idx], probs[idx]))
        elif self.data_name == "MAG":
            idx = labels != len(np.unique(labels)) - 1
            self.metrics[name.lower() + "_bal_accuracy"].append(
                balanced_accuracy_score(labels[idx], predicted[idx])
            )
            self.metrics[name.lower() + "_f1_score"].append(
                f1_score(
                    labels,
                    predicted,
                    average="macro" if probs.shape[1] > 2 else "binary",
                )
            )                                    
        else:
            self.metrics[name.lower() + "_bal_accuracy"].append(
                balanced_accuracy_score(labels, predicted)
            )
            self.metrics[name.lower() + "_f1_score"].append(
                f1_score(
                    labels,
                    predicted,
                    average="macro" if probs.shape[1] > 2 else "binary",
                )
            )
        # pdb.set_trace()
        # self.metrics[name.lower() + "_auc"].append(
        #     roc_auc_score(
        #         labels,
        #         probs_for_auc,
        #         multi_class="ovr" if probs.shape[1] > 2 else "raise",
        #     )
        # )
        ece = compute_ece(labels, probs, n_bins=10)
        self.metrics[name.lower() + "_ece"].append(ece)

        results = f"{name.lower()} : "
        for metric in ["accuracy", "bal_accuracy", "f1_score", "ece"]:
            metric_name = f"{name.lower()}_{metric}"
            results += f"{metric}: {self.metrics[metric_name][-1]*100:.2f} "
        logging.info(results)

        if name not in ["Adapted_val", "Val"] and self.data_name != "Arxiv":
            self.metrics[name.lower() + "_cls_accuracy"].append(
                recall_score(labels, predicted, average=None)
            )
            # print(self.metrics[name.lower() + "_cls_accuracy"][-1].shape)
            classwise_acc = self.metrics[name.lower() + "_cls_accuracy"][-1]
            classwise_results = f"{name.lower()} class-wise accuracy:\n"
            for idx, acc in enumerate(classwise_acc):
                classwise_results += f"Class {idx}: {acc*100:.2f} "
            logging.info(classwise_results)

    def save_time(self, name, time):
        self.metrics[name].append(time)

    def summarize_results(self):
        for key, values in self.metrics.items():
            if isinstance(values[0], np.ndarray):
                # Handle class-wise accuracy separately
                class_values = np.stack(values)
                mean = np.mean(class_values, axis=0)
                std = np.std(class_values, axis=0)
                class_results = " ".join(
                    [
                        f"Class{i}:{mean[i]*100:.2f}$\pm${std[i]*100:.2f}"
                        for i in range(len(mean))
                    ]
                )
                logging.info(f"{key}: {class_results}")
            elif "time" in key.split("_"):
                mean = np.mean(values)
                std = np.std(values)
                logging.info(f"{key}: {mean:.3f}$\pm${std:.3f}")                
            else:
                mean = np.mean(values)
                std = np.std(values)
                logging.info(f"{key}: {mean*100:.2f}$\pm${std*100:.2f}")

    def save_to_csv(self, filename="metrics.csv"):
        if not self.metrics:
            self.calculate_metrics()

        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics.keys())
            writer.writeheader()
            writer.writerow(self.metrics)

    def save_to_json(self, filename="metrics.json"):
        if not self.metrics:
            self.calculate_metrics()

        with open(filename, "w") as file:
            json.dump(self.metrics, file)

    def save_to_dataframe(self, filename="metrics.csv"):
        if not self.metrics:
            self.calculate_metrics()

        df = pd.DataFrame([self.metrics])
        df.to_csv(filename, index=False)


def compute_ece(labels, probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (conf >= bin_lower) & (conf < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (labels[in_bin] == pred[in_bin]).mean()
            avg_confidence_in_bin = conf[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def rank_accuracy(labels, probs):
    class1_scores = probs[:, 1]
    sorted_indices = np.argsort(class1_scores)[::-1]
    # Apply the threshold to determine class predictions
    threshold_index = np.bincount(labels)[-1]
    ranked_labels = np.zeros(len(class1_scores), dtype=int)
    ranked_labels[sorted_indices[:threshold_index]] = 1  # Set top samples to class 1

    # Calculate accuracy by comparing predicted labels with true labels
    correct_predictions = np.sum(ranked_labels == labels)
    return correct_predictions / len(labels)