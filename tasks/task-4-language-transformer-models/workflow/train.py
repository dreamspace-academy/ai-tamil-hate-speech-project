import os
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from data.helpers import get_data_loaders
from models.transformer import SentClf
from utils.logger import create_logger
from utils.utils import *


def get_args():
    with open(r'params.yaml') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    args = Namespace(**args)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.savedir = os.path.join(os.getcwd(), "model_artifacts")
    #args.savedir = os.path.join(savedir, args.model_path)
    os.makedirs(args.savedir, exist_ok=True)

    return args


def get_criterion(args):
    freqs = [args.label_freqs[l] for l in args.labels]
    label_weights = (np.array(freqs) / args.train_data_len) ** -1

    if args.loss_type == 'focal_ce':
        if args.num_labels == 2:
            cross_entropy_fn = nn.BCEWithLogitsLoss()
        else:
            cross_entropy_fn = nn.CrossEntropyLoss()
        criterion = FocalLoss(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            cross_entropy_fn=cross_entropy_fn,
            num_labels=args.num_labels,
        )
    elif args.loss_type == 'smoothed_focal_ce' and args.num_labels > 2:
        cross_entropy_fn = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing,
            size_average=None, 
            ignore_index=-100,
            reduction='mean',
        )
        criterion = FocalLoss(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            cross_entropy_fn=cross_entropy_fn,
            num_labels=args.num_labels,
        )
    elif args.loss_type == 'weighted_ce':
        if args.num_labels == 2:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([label_weights[1]]).to(args.device))
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(label_weights).to(args.device))
    elif args.loss_type == 'smoothed_weighted_ce' and args.num_labels > 2:
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(label_weights).to(args.device),
            size_average=None, 
            ignore_index=-100,
            reduction='mean',
            label_smoothing=args.label_smoothing
        )
    else:
        if args.num_labels == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
    
    return criterion


def get_optimizer(model, args):
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
    ]

    return optim.AdamW(optimizer_grouped_parameters, lr=args.lr)


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor)


def model_eval(i_epoch, data, data_partition_name, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, prob_preds, tgts = [], [], [], []
        for batch in data:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

            if args.num_labels == 2:
                out_probs = torch.sigmoid(out).cpu().detach().numpy()
                prob_preds.append(out_probs)
                pred = np.where(out_probs > 0.5, 1, 0)
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            preds.append(pred)
            
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    precision_recall_f1score_metric = precision_recall_fscore_support(tgts, preds, average='macro')
    metrics["Accuracy"] = accuracy_score(tgts, preds)
    metrics["Precision"] = precision_recall_f1score_metric[0]
    metrics["Recall"] = precision_recall_f1score_metric[1]
    metrics["F1"] = precision_recall_f1score_metric[2]
    
    if store_preds:
        if args.num_labels == 2:
            store_preds_to_disk(tgts, preds, prob_preds, args)
        else:
            store_preds_to_disk(tgts, preds, args)

    return metrics


def model_forward(i_epoch, model, args, criterion, batch):
    
    device = next(model.parameters()).device
    
    txt, segment, mask, tgt = batch[0], batch[1], batch[2], batch[3]
    txt, mask, segment = txt.to(device), mask.to(device), segment.to(device)
    tgt = tgt.to(device)

    if args.use_fp16:
        with torch.cuda.amp.autocast():
            out = model(txt, mask, segment)
            if args.num_labels == 2:
                loss = criterion(out, torch.unsqueeze(tgt.type_as(out), 1))
            else:
                loss = criterion(out, tgt)
    else:
        out = model(txt, mask, segment)
        if args.num_labels == 2:
            loss = criterion(out, torch.unsqueeze(tgt.type_as(out), 1))
        else:
            loss = criterion(out, tgt)

    return loss, out, tgt


def train(args):
    set_seed(args.seed)
    set_mlflow(args)
    mlflow.log_params(namespace_to_dict(args))
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    cuda_len = torch.cuda.device_count()

    train_loader, val_loader, test_loader = get_data_loaders(args)
    model = SentClf(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger.info(model)

    if cuda_len > 1:
        logger.info('Model in parallel')
        model = nn.DataParallel(model)

    model.to(args.device)

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    validation_metrics_history = {
        "loss": [],
        "F1": [],
        "Precision": [],
        "Recall": [],
        "Accuracy": []
    }
    train_loss = []

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model_dict = model.state_dict()
        model_dict.update(checkpoint["state_dict"]) 
        model.load_state_dict(model_dict)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    logger.info("Training..")
    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _ = model_forward(i_epoch, model, args, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            train_losses.append(loss.item())

            if args.use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                if args.use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(i_epoch, val_loader, "Validation", model, args, criterion)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", global_step, metrics, args, logger)
        train_loss.append(np.mean(train_losses))
        append_metrics(validation_metrics_history, metrics)

        tuning_metric = metrics[args.tunning_metric]
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
            logger=logger
        )

        if n_no_improve >= args.es_patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()

    test_metrics = model_eval(np.inf, test_loader, "Test", model, args, criterion, store_preds=True)
    log_metrics(f"Test", global_step, test_metrics, args, logger)

    mlflow.log_artifact(f"{args.savedir}/logfile.log")
    mlflow.log_artifact(f"{args.savedir}/test_labels_pred.txt")
    mlflow.log_artifact(f"{args.savedir}/test_labels_prob_pred.txt")
    mlflow.log_artifact(f"{args.savedir}/test_labels_gold.txt")

    mlflow.end_run()
    

if __name__ == "__main__":
    args = get_args()
    train(args)