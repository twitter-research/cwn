import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from sklearn import metrics as met
from data.complex import ComplexBatch
from ogb.graphproppred import Evaluator as OGBEvaluator

cls_criterion = torch.nn.CrossEntropyLoss()
bicls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.L1Loss()
msereg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type='classification', ignore_unlabeled=False):
    """
        Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """

    if task_type == 'classification':
        loss_fn = cls_criterion
    elif task_type == 'bin_classification':
        loss_fn = bicls_criterion
    elif task_type == 'regression':
        loss_fn = reg_criterion
    elif task_type == 'mse_regression':
        loss_fn = msereg_criterion
    else:
        raise NotImplementedError('Training on task type {} not yet supported.'.format(task_type))
    
    curve = list()
    model.train()
    num_skips = 0
    for step, batch in enumerate(tqdm(loader, desc="Training iteration")):
        batch = batch.to(device)
        if isinstance(batch, ComplexBatch):
            num_samples = batch.cochains[0].x.size(0)
            for dim in range(1, batch.dimension+1):
                num_samples = min(num_samples, batch.cochains[dim].num_cells)
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
        if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
            targets = batch.y.view(-1,)
        else:
            targets = batch.y.to(torch.float32).view(pred.shape)

        # In some ogbg-mol* datasets we may have null targets.
        # When the cross entropy loss is used and targets are of shape (N,)
        # the maks is broadcasted automatically to the shape of the predictions.
        mask = ~torch.isnan(targets)
        loss = loss_fn(pred[mask], targets[mask])

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


def eval(model, device, loader, evaluator, task_type):
    """
        Evaluates a model over all the batches of a data loader.
    """

    if task_type == 'classification':
        loss_fn = cls_criterion
    elif task_type == 'bin_classification':
        loss_fn = bicls_criterion
    elif task_type == 'regression':
        loss_fn = reg_criterion
    elif task_type == 'mse_regression':
        loss_fn = msereg_criterion
    else:
        loss_fn = None
            
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="Eval iteration")):
        
        # Cast features to double precision if that is used
        if torch.get_default_dtype() == torch.float64:
            for dim in range(batch.dimension + 1):
                batch.cochains[dim].x = batch.cochains[dim].x.double()
                assert batch.cochains[dim].x.dtype == torch.float64, batch.cochains[dim].x.dtype

        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            
            if task_type != 'isomorphism':
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    targets = batch.y.view(-1,)
                    y_true.append(batch.y.detach().cpu())
                else:
                    targets = batch.y.to(torch.float32).view(pred.shape)
                    y_true.append(batch.y.view(pred.shape).detach().cpu())
                mask = ~torch.isnan(targets)  # In some ogbg-mol* datasets we may have null targets.
                loss = loss_fn(pred[mask], targets[mask])
                losses.append(loss.detach().cpu().item())
            else:
                assert loss_fn is None
            
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()  if len(y_true) > 0 else None
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {'y_pred': y_pred, 'y_true': y_true}
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan
    return evaluator.eval(input_dict), mean_loss

    
class Evaluator(object):
    
    def __init__(self, metric, **kwargs):
        if metric == 'isomorphism':
            self.eval_fn = self._isomorphism
            self.eps = kwargs.get('eps', 0.01)
            self.p_norm = kwargs.get('p', 2)
        elif metric == 'accuracy':
            self.eval_fn = self._accuracy
        elif metric == 'mae':
            self.eval_fn = self._mae
        elif metric.startswith('ogbg-mol'):
            self._ogb_evaluator = OGBEvaluator(metric)
            self._key = self._ogb_evaluator.eval_metric
            self.eval_fn = self._ogb
        else:
            raise NotImplementedError('Metric {} is not yet supported.'.format(metric))
    
    def eval(self, input_dict):
        return self.eval_fn(input_dict)
        
    def _isomorphism(self, input_dict):
        # NB: here we return the failure percentage... the smaller the better!
        preds = input_dict['y_pred']
        assert preds is not None
        assert preds.dtype == np.float64
        preds = torch.tensor(preds, dtype=torch.float64)
        mm = torch.pdist(preds, p=self.p_norm)
        wrong = (mm < self.eps).sum().item()
        metric = wrong / mm.shape[0]
        return metric
    
    def _accuracy(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = np.argmax(input_dict['y_pred'], axis=1)
        assert y_true is not None
        assert y_pred is not None
        metric = met.accuracy_score(y_true, y_pred)
        return metric

    def _mae(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        assert y_true is not None
        assert y_pred is not None
        metric = met.mean_absolute_error(y_true, y_pred)
        return metric
    
    def _ogb(self, input_dict, **kwargs):
        assert 'y_true' in input_dict
        assert input_dict['y_true'] is not None
        assert 'y_pred' in input_dict
        assert input_dict['y_pred'] is not None
        return self._ogb_evaluator.eval(input_dict)[self._key]
