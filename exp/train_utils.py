import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from sklearn import metrics as met
from data.complex import ComplexBatch

cls_criterion = torch.nn.CrossEntropyLoss()
reg_criterion = torch.nn.L1Loss()


def train(model, device, loader, optimizer, task_type='classification', ignore_unlabeled=False):
    """
        Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """

    if task_type == 'classification':
        loss_fn = cls_criterion
    elif task_type == 'regression':
        loss_fn = reg_criterion
    else:
        raise NotImplementedError('Training on task type {} not yet supported.'.format(task_type))
    
    curve = list()
    model.train()
    num_skips = 0
    for step, batch in enumerate(tqdm(loader, desc="Training iteration")):
        batch = batch.to(device)
        if isinstance(batch, ComplexBatch):
            num_samples = batch.chains[0].x.size(0)
            for dim in range(1, batch.dimension+1):
                num_samples = min(num_samples, batch.chains[dim].num_simplices)
        else:
            # This is graph.
            num_samples = batch.x.size(0)

        if num_samples <= 1:
            # Skip batch if it only comprises one sample (could cause problems with BN)
            num_skips += 1
            if float(num_skips) / len(loader) >= 0.25:
                logging.warning("Warning! 25% of the batches were skipped this epoch")
            continue
        
        # (DEBUG)
        if num_samples < 10:
            logging.warning("Warning! BatchNorm applied on a batch "
                            "with only {} samples".format(num_samples))

        optimizer.zero_grad()
        pred = model(batch)
        if task_type == 'regression':
            loss = loss_fn(pred, batch.y.view(-1, 1))
        else:
            loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        curve.append(loss.detach().cpu().item())

    return curve


def infer(model, device, loader):
    """
        Runs inference over all the batches of a data loader.
    """
    model.eval()
    y_pred = list()
    for step, batch in enumerate(tqdm(loader, desc="Inference iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y_pred.append(pred.detach().cpu())
    y_pred = torch.cat(y_pred, dim=0).numpy()
    return y_pred


def eval(model, device, loader, evaluator, task_type, debug_dataset=None):
    """
        Evaluates a model over all the batches of a data loader.
    """

    if task_type == 'classification':
        loss_fn = cls_criterion
    elif task_type == 'regression':
        loss_fn = reg_criterion
    else:
        loss_fn = None
    
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="Eval iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            if task_type == 'regression':
                loss = loss_fn(pred, batch.y.view(-1, 1))
            else:
                loss = loss_fn(pred, batch.y.view(-1))
        losses.append(loss.detach().cpu().item())

        y_true.append(batch.y.detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {'y_pred': y_pred, 'y_true': y_true}
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan
    return evaluator.eval(input_dict), mean_loss

    
class Evaluator(object):
    
    def __init__(self, metric):
        if metric == 'isomorphism':
            self.eval_fn = self._isomorphism
        elif metric == 'accuracy':
            self.eval_fn = self._accuracy
        elif metric == 'mae':
            self.eval_fn = self._mae
        else:
            raise NotImplementedError('Metric {} is not yet supported.'.format(metric))
    
    def eval(self, input_dict):
        return self.eval_fn(input_dict)
        
    def _isomorphism(self, input_dict):
        # NB: here we return the failure percentage... the smaller the better!
        p = input_dict.get('p', 2)
        eps = input_dict.get('eps', 0.01)
        preds = input_dict['y_pred']
        mm = torch.pdist(torch.tensor(preds, dtype=torch.float32), p=p)
        wrong = (mm < eps).sum().item()
        metric = wrong / mm.shape[0]
        return metric
    
    def _accuracy(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = np.argmax(input_dict['y_pred'], axis=1)
        metric = met.accuracy_score(y_true, y_pred)
        return metric

    def _mae(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        metric = met.mean_absolute_error(y_true, y_pred)
        return metric
