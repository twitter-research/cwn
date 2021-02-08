import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics as met
from data.complex import ComplexBatch

# cls_criterion = torch.nn.BCEWithLogitsLoss()
cls_criterion = torch.nn.CrossEntropyLoss()
reg_criterion = torch.nn.MSELoss()


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
    for step, batch in enumerate(tqdm(loader, desc="Training iteration")):
        batch = batch.to(device)
        num_samples = batch.chains[0].x.size(0)
        for dim in range(1, batch.dimension+1):
            num_samples = min(num_samples, batch.chains[dim].x.size(0))

        if num_samples <= 1:
            # Skip batch if it only comprises one sample (could cause problems with BN)
            continue

        optimizer.zero_grad()
        pred = model(batch)
        # TODO: shall we do some dtype checking here on the y?
        if ignore_unlabeled:
            is_labeled = batch.y == batch.y
            loss = loss_fn(pred[is_labeled], batch.y[is_labeled])
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
        raise NotImplementedError('Training on task type {} not yet supported.'.format(task_type))
    
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    for step, batch in enumerate(tqdm(loader, desc="Eval iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
        losses.append(loss.detach().cpu().item())

        y_true.append(batch.y.detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # # Test the predictions are the same without batching
    # if dataset is not None:
    #     y_true2 = []
    #     y_pred2 = []
    #     for step, batch in enumerate(tqdm(dataset, desc="Eval assert iteration")):
    #         batch = ComplexBatch.from_complex_list([batch])
    #         batch = batch.to(device)
    #         with torch.no_grad():
    #             pred = model(batch)
    #
    #         y_true2.append(batch.y.detach().cpu())
    #         y_pred2.append(pred.detach().cpu())
    #
    #     y_true2 = torch.cat(y_true2, dim=0).numpy()
    #     y_pred2 = torch.cat(y_pred2, dim=0).numpy()
    #
    #     assert np.array_equal(y_true, y_true2), print(y_true, y_true2)
    #     assert np.allclose(y_pred, y_pred2), print(np.abs(y_pred -y_pred2))

    input_dict = {'y_pred': y_pred, 'y_true': y_true}
    return evaluator.eval(input_dict), float(np.mean(losses))

    
class Evaluator(object):
    
    def __init__(self, metric):
        if metric == 'isomorphism':
            self.eval_fn = self._isomorphism
        elif metric == 'accuracy':
            self.eval_fn = self._accuracy
        else:
            raise NotImplementedError('Metric {} is not yet supported.'.format(metric))
    
    def eval(self, input_dict):
        return self.eval_fn(input_dict)
        
    def _isomorphism(self, input_dict, **kwargs):
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
