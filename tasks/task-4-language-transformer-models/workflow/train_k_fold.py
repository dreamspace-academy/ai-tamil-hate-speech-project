import os
import json
from argparse import Namespace
from tqdm import tqdm
import numpy as np
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve
)

from data.helpers import get_data_loaders, load_dataset
from models.transformer import SentClf
from utils.logger import create_logger
from utils.utils import *


def get_args():
    with open(r'params.yaml') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    args = Namespace(**args)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.savedir = os.path.join(os.getcwd(), "model_artifacts")
    os.makedirs(args.savedir, exist_ok=True)

    return args


def get_criterion(args):
    freqs = [args.label_freqs[l] for l in args.labels]
    label_weights = (np.array(freqs) / args.train_data_len) ** -1

    if args.loss_type == 'weighted_ce':
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


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, prob_preds, tgts = [], [], [], []
        for batch in data:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

            if args.num_labels == 2:
                out_probs = torch.sigmoid(out).cpu().detach().numpy()
                prob_preds.extend(out_probs[:,0].tolist())
                pred = np.where(out_probs > args.class_threshold, 1, 0)
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
            display = PrecisionRecallDisplay.from_predictions(y_true=tgts, y_pred=prob_preds)
            display.figure_.savefig(os.path.join(args.savedir, 'pr_curve.png'))
            pr_values = precision_recall_curve(y_true=tgts, probas_pred=prob_preds)
            store_preds_to_disk(args, tgts, preds, prob_preds, pr_values)
        else:
            store_preds_to_disk(args, tgts, preds)

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

    metrics_fold = {
        "Accuracy": [],
        "F1": [],
        "Precision": [],
        "Recall": [],
    }

    data = load_dataset(args)
    skf = StratifiedKFold(n_splits=args.number_folds, random_state=args.seed, shuffle=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    start_time = time.time()
    for k, (train_index, dev_index) in enumerate(skf.split(data, data.label)):

        train_corpus, dev_corpus = data.iloc[train_index], data.iloc[dev_index]
        train_loader, val_loader = get_data_loaders(args, train_corpus, dev_corpus, tokenizer)
        
        model = SentClf(args)
        criterion = get_criterion(args)
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

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

        if args.use_fp16:
            scaler = torch.cuda.amp.GradScaler()

        logger.info(f"Training {k+1}th fold..")
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
            metrics = model_eval(i_epoch, val_loader, model, args, criterion)
            logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
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
                model.state_dict(),
                is_improvement,
                args.savedir,
                logger=logger
            )

            if n_no_improve >= args.es_patience:
                logger.info("No improvement. Breaking out of loop.")
                break

        load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
        model.eval()
        test_metrics = model_eval(np.inf, val_loader, model, args, criterion, store_preds=True)
        log_metrics(f"Validation {k+1}th fold", global_step, test_metrics, args, logger)

        metrics_fold["Accuracy"].append(test_metrics['Accuracy'])
        metrics_fold["F1"].append(test_metrics['F1'])
        metrics_fold["Precision"].append(test_metrics['Precision'])
        metrics_fold["Recall"].append(test_metrics['Recall'])

        os.remove(os.path.join(args.savedir, "model_best.pt"))
        os.remove(os.path.join(args.savedir, "checkpoint.pt"))
    
    end_time = time.time()
    
    accuracy_mean = np.mean(metrics_fold['Accuracy'])
    accuracy_std = np.std(metrics_fold['Accuracy'])
    f1_mean = np.mean(metrics_fold['F1'])
    f1_std = np.std(metrics_fold['F1'])
    precision_mean = np.mean(metrics_fold['Precision'])
    precision_std = np.std(metrics_fold['Precision'])
    recall_mean = np.mean(metrics_fold['Recall'])
    recall_std = np.std(metrics_fold['Recall'])

    logger.info(f"Cross validation Accuracy: {accuracy_mean} ± {accuracy_std}")
    logger.info(f"Cross validation F1: {f1_mean} ± {f1_std}")
    logger.info(f"Cross validation Precision: {precision_mean} ± {precision_std}")
    logger.info(f"Cross validation Recall: {recall_mean} ± {recall_std}")

    # save k-fold cross-validation metrics
    k_fold_cv_metrics = {
        'accuracy': {
            'mean': accuracy_mean,
            'std': accuracy_std,
        },
        'f1': {
            'mean': f1_mean,
            'std': f1_std,
        },
        'precision': {
            'mean': precision_mean,
            'std': precision_std,
        },
        'recall': {
            'mean': recall_mean,
            'std': recall_std,
        },
        'elapsed_time': f"{(end_time-start_time):.2f} s",
    }

    with open('cv_results.json', 'w') as fp:
        json.dump(k_fold_cv_metrics, fp)

    mlflow.end_run()
    

if __name__ == "__main__":
    args = get_args()
    train(args)