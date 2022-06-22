import contextlib
import numpy as np
import random
import shutil
import os
import mlflow
from argparse import Namespace

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, cross_entropy_fn=None, num_labels=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy_fn = cross_entropy_fn
        self.num_labels = num_labels

    def forward(self, inputs, targets):
        if self.num_labels == 2:
            ce_loss = self.bce(inputs, targets)
        else:
            ce_loss = self.cross_entropy_fn(inputs,  targets)
        loss = self.alpha * (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        return loss


def set_mlflow(args):
    MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
    MLFLOW_TRACKING_USERNAME = os.environ['MLFLOW_TRACKING_USERNAME']
    MLFLOW_TRACKING_PASSWORD = os.environ['MLFLOW_TRACKING_PASSWORD']

    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.start_run()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, Namespace) else v
        for k, v in vars(namespace).items()
    }


def store_preds_to_disk(args, tgts, preds, prob_preds=None, pr_values=None):
    with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
        fw.write("\n".join([str(x) for x in preds]))
    if prob_preds:
        with open(os.path.join(args.savedir, "test_labels_prob_pred.txt"), "w") as fw:
            fw.write("\n".join([str(x) for x in prob_preds]))
    if pr_values:
        with open(os.path.join(args.savedir, "test_pr_values.txt"), "w") as fw:
            fw.write("precision,recall,threshold\n")
            fw.write("\n".join([f"{p},{r},{thr}" for p,r,thr in zip(pr_values[0], pr_values[1], pr_values[2])]))
    with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
        fw.write("\n".join([str(x) for x in tgts]))
    with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
        fw.write(" ".join([str(l) for l in args.labels]))


def log_metrics(set_name, step, metrics, args, logger):
    logger.info(
        "{}: Loss: {:.5f} | F1: {:.5f} | Precision: {:.5f} | Recall: {:.5f} | Accuracy: {:.5f}".format(
            set_name,
            metrics["loss"],
            metrics["F1"],
            metrics["Precision"],
            metrics["Recall"],
            metrics["Accuracy"]
        )
    )
    mlflow.log_metric(f"{set_name} Accuracy", metrics["Accuracy"], step)
    mlflow.log_metric(f"{set_name} Precision", metrics["Precision"], step)
    mlflow.log_metric(f"{set_name} Recall", metrics["Recall"], step)
    mlflow.log_metric(f"{set_name} F1", metrics["F1"], step)
    mlflow.log_metric(f"{set_name} Loss", metrics["loss"], step)


def append_metrics(metrics_history, metrics):
    for metric_name in metrics_history.keys():
        metrics_history[metric_name].append(metrics[metric_name])


def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt", logger=None):
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)


def load_labels(path):
    with open(path, "r") as f:
        all_preds = f.readlines()
    preds_processed = [[int(e) for e in pred.strip('\n').split()] for pred in all_preds]
    return np.array(preds_processed)