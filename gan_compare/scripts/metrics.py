import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def calc_all_scores(y_true, y_prob_logit, v_loss, run_type, epoch=None):
    """Calculates, outputs, and returns loss, accuracy, precision, recall, F1-score, ROC_AUC, and PRC_AUC

    Args:
        y_true ([type]): labels for each sample and each class (N x C)
        y_prob_logit ([type]): log-probabilities for each sample and each class (N x C)
        v_loss ([type]): loss of each sample (N)
        run_type (String): either 'train', 'val', or 'test'
        epoch (int, optional): which epoch number to print. Defaults to None.

    Returns:
        [type]: [description]
    """
    y_prob = torch.exp(y_prob_logit)  # probabilities of all classes for each sample
    _, y_pred = torch.max(y_prob, 1)  # class prediction for each sample
    loss = calc_loss(v_loss, run_type, epoch)
    acc_score = calc_accuracy(y_true, y_pred, run_type, epoch)
    prec_rec_f1_scores = calc_prec_rec_f1_scores(y_true, y_pred, run_type, epoch)
    # TODO calculating prec, rec, F1, AUROC, AURPC may be only applicable if classying binary?
    roc_auc = calc_AUROC(y_true, y_prob, run_type, epoch)
    prc_auc = calc_AUPRC(y_true, y_prob, run_type, epoch)
    return loss, acc_score, prec_rec_f1_scores, roc_auc, prc_auc


def calc_loss(v_loss, run_type, epoch=None):
    loss = np.mean(v_loss)
    if epoch is None:
        logging.info(f"{run_type} loss: {loss}")
    else:
        logging.info(f"{run_type} loss in {epoch} epoch: {loss}")
    return loss


def calc_accuracy(y_true, y_pred, run_type, epoch=None):
    score = accuracy_score(y_true, y_pred)
    if epoch is None:
        logging.info(f"{run_type} accuracy:  {score}")
    else:
        logging.info(f"{run_type} accuracy in {epoch} epoch:  {score}")
    return score


def calc_prec_rec_f1_scores(y_true, y_pred, run_type, epoch=None):
    prec, rec, f1, _ = np.array(precision_recall_fscore_support(y_true, y_pred))[
        :, -1
    ]  # keep scores for label==1

    epoch_text = "" if epoch is None else f" in {epoch} epoch"

    logging.info(f"{run_type} Precision{epoch_text}: {prec}")
    logging.info(f"{run_type} Recall{epoch_text}:    {rec}")
    logging.info(f"{run_type} F1-Score{epoch_text}:  {f1}")

    return prec, rec, f1


def calc_AUROC(y_true, y_prob, run_type, epoch=None):
    try:
        roc_auc = roc_auc_score(y_true, y_prob[:, -1])
        if epoch is None:
            logging.info(f"{run_type} AUROC: {roc_auc}")
        else:
            logging.info(f"{run_type} AUROC in {epoch} epoch: {roc_auc}")
        return roc_auc
    except ValueError:
        logging.warning(
            f"WARNING cannot calculate AUROC. calc_AUROC got a ValueError. This happens for example when only one class is present in the dataset; run_type: {run_type}; epoch: {epoch}"
        )
    except IndexError:
        logging.warning(
            f"WARNING cannot calculate AUROC. calc_AUROC got an IndexError. This happens for example when the respective dataset is empty, for example when filtering for calcifications although there are no calcifications; run_type: {run_type}; epoch: {epoch}"
        )


def calc_AUPRC(y_true, y_prob, run_type, epoch=None):
    try:
        prc_auc = average_precision_score(y_true, y_prob[:, -1])
        if epoch is None:
            logging.info(f"{run_type} AUPRC: {prc_auc}")
        else:
            logging.info(f"{run_type} AUPRC in {epoch} epoch: {prc_auc}")
        return prc_auc
    except ValueError:
        logging.warning(
            f"WARNING cannot calculate AUPRC. calc_AUPRC got a ValueError. This happens for example when only one class is present in the dataset;; run_type: {run_type}; epoch: {epoch}"
        )
    except IndexError:
        logging.warning(
            f"WARNING cannot calculate AUPRC. calc_AUPRC got an IndexError. This happens for example when the respective dataset is empty, for example when filtering for calcifications although there are no calcifications; run_type: {run_type}; epoch: {epoch}"
        )


def output_ROC_curve(y_true, y_prob_logit, run_type, logfilename):
    """Only for binary classification

    Args:
        y_true ([type]): [description]
        y_prob_logit ([type]): [description]
        run_type ([type]): [description]
        logfilename ([type]): location where to save the output figure
    """
    y_prob = torch.exp(y_prob_logit)[
        :, -1
    ]  # keep only the probabilities for the true class

    # Calculate ROC values:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC values:
    plt.figure(figsize=(6, 6))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{run_type} set; receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(f"ROC_curve_{logfilename}.png")
