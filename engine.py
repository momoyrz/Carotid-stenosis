# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import math
import sys
from typing import Iterable, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from sklearn.manifold import TSNE
from utility import utils
from utility.preprocessing import sparse_to_tensor
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
    cohen_kappa_score


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        samples, targets, adj, k = data['image'].cuda(), data['label'].cuda(), data['adj'], data['k']
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        adj_mats2 = {key: sparse_to_tensor(value).to_dense()[k:].cuda() for key, value in adj.items()}

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():

                output = model(samples, adj_mats2, k=k)
                loss = criterion(output, targets[k:])
        else:  # full precision
            output = model(samples, adj_mats2, k=k)
            loss = criterion(output, targets[k:])

        loss_value = loss.item()

        if not math.isfinite(loss_value):  # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:  # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets[k:]).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        images, target, adj, k = batch['image'].cuda(), batch['label'].cuda(), batch['adj'], batch['k']

        adj_mats2 = {key: sparse_to_tensor(value).to_dense()[k:].cuda() for key, value in adj.items()}
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images, adj_mats2, k=k)
                loss = criterion(output, target[k:])
        else:
            output = model(images, adj_mats2, k=k)
            loss = criterion(output, target[k:])

        acc1, acc5 = accuracy(output, target[k:], topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def my_evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    y_true, y_pred = [], []
    y_score = pd.DataFrame()
    features = torch.Tensor().to(device)
    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images, target, adj, k = batch['image'].cuda(), batch['label'].cuda(), batch['adj'], batch['k']

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        adj_mats2 = {key: sparse_to_tensor(value).to_dense()[k:].cuda() for key, value in adj.items()}

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images, adj_mats2, k=k)
                loss = criterion(output, target[k:])
        else:
            output = model(images, adj_mats2, k=k)
            loss = criterion(output, target[k:])

        acc1, acc5 = accuracy(output, target[k:], topk=(1, 2))

        # calculate predictions and true values
        _, pred = torch.max(output, dim=1)
        y_true += target[k:].cpu().numpy().tolist()
        y_pred += pred.cpu().numpy().tolist()
        y_score = y_score.append(output.cpu().numpy().tolist())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # calculate and print additional evaluation metrics

    report = classification_report(y_true, y_pred, output_dict=True)
    macro_auc, macro_pre, macro_sen, macro_f1, macro_spec, kappa = [], [], [], [], [], []
    for i in range(len(report) - 3):
        fpr, tpr, thresholds = roc_curve(y_true, list(y_score.iloc[:, i]), pos_label=i)
        roc_auc = auc(fpr, tpr)
        macro_auc.append(roc_auc)
        pre = report[str(i)]['precision']
        sen = report[str(i)]['recall']
        f1 = report[str(i)]['f1-score']
        macro_pre.append(pre)
        macro_sen.append(sen)
        macro_f1.append(f1)
        kappa.append(report[str(i)]['support'] * report[str(i)]['f1-score'])
        # plot ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for class %d' % i)
        plt.legend(loc="lower right")
        plt.show()

    # calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(data_loader.dataset.classes))
    plt.xticks(tick_marks, data_loader.dataset.classes, rotation=45)
    plt.yticks(tick_marks, data_loader.dataset.classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print('Macro AUC: {:.3f}'.format(np.mean(macro_auc)))
    print('Macro Pre: {:.3f}'.format(np.mean(macro_pre)))
    print('Macro Sen: {:.3f}'.format(np.mean(macro_sen)))
    print('Macro F1: {:.3f}'.format(np.mean(macro_f1)))
    # print('Macro Spec: {:.3f}'.format(np.mean(macro_spec)))
    print('Kappa: {:.3f}'.format(sum(kappa) / len(y_true)))

    features = features.detach().cpu().numpy()
    Tsne = TSNE(n_components=2, perplexity=50, init='pca', random_state=0)
    x_tsne = Tsne.fit_transform(features, y_true)
    n_samples = len(y_true)
    colors = ['r', 'b', 'g']

    plt.clf()
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.scatter(x_tsne[i, 0], x_tsne[i, 1], c=colors[y_true[i]])
    plt.show()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate1(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluate mode
    model.eval()
    y_true, y_pred, y_score = [], [], pd.DataFrame()
    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images, target, adj, k = batch['image'].cuda(), batch['label'].cuda(), batch['adj'], batch['k']
        adj_mats2 = {key: sparse_to_tensor(value).to_dense()[k:].cuda() for key, value in adj.items()}
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images, adj_mats2, k=k)
                loss = criterion(output, target[k:])
        else:
            output = model(images, adj_mats2, k=k)
            loss = criterion(output, target[k:])

        # calculate predictions and true values
        y_true += target[k:].cpu().numpy().tolist()
        y_pred += output.argmax(1).cpu().numpy().tolist()
        y_score = pd.concat([y_score, pd.DataFrame(output.cpu().numpy())], axis=0)

    acc, pre, sen, f1, spec, kappa = accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average='macro'), \
        recall_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='macro'), \
        calculate_average_specificity_sklearn(y_true, y_pred), cohen_kappa_score(y_true, y_pred)

    acc_plot_confusion_matrix(y_true, y_pred, data_loader)

    # plot ROC curve
    auc_list = []
    for i in range(len(data_loader.dataset.classes)):
        fpr, tpr, _ = roc_curve(y_true, y_score[i], pos_label=i)
        roc_data = np.column_stack((fpr, tpr))
        np.savetxt('./results/roc_data_{}.txt'.format(data_loader.dataset.classes[i]), roc_data, delimiter=',')
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of {}'.format(data_loader.dataset.classes[i]))
        plt.legend(loc="lower right")
        plt.show()

    return {'acc': acc, 'pre': pre, 'sen': sen, 'f1': f1, 'spec': spec, 'kappa': kappa,
            'auc': sum(auc_list) / len(auc_list)}


def acc_plot_confusion_matrix(y_true, y_pred, data_loader):
    # calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # 保存混淆矩阵
    np.savetxt('./results/confusion_matrix.txt', cm, fmt='%d', delimiter=',')
    plt.figure(figsize=(5, 5), dpi=960)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(data_loader.dataset.classes))
    plt.xticks(tick_marks, data_loader.dataset.classes)
    plt.yticks(tick_marks, data_loader.dataset.classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./results/confusion_matrix.png', dpi=960)
    plt.show()


def calculate_specificity_sklearn(cm, class_index):
    true_negatives = np.sum(np.delete(np.delete(cm, class_index, axis=0), class_index, axis=1))
    false_positives = np.sum(np.delete(cm, class_index, axis=0)[:, class_index])
    specificity = true_negatives / (true_negatives + false_positives)
    return specificity


def calculate_average_specificity_sklearn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificities = []

    for class_index in range(num_classes):
        specificity = calculate_specificity_sklearn(cm, class_index)
        specificities.append(specificity)

    average_specificity = np.mean(specificities)
    return average_specificity