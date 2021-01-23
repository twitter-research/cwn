import os
import torch
from tqdm import tqdm
from definitions import ROOT_DIR
from sklearn import metrics as met
from data.dataset import SRDataset

cls_criterion = torch.nn.BCEWithLogitsLoss()
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
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            # TODO: figure out why this exactly (it seems like it drops batches with only one sample)
            # NB: this routine is heavily inspired by:
            #     https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            if ignore_unlabeled:
                is_labeled = batch.y == batch.y
                loss = loss_fn(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = loss_fn(pred.to(torch.float32), batch.y.to(torch.float32))
            loss.backward()
            optimizer.step()
            curve.append(loss.detach().numpy())
            
    return curve

def infer(model, device, loader):
    """
        Runs inference over all the batches of a data loader.
    """
    model.eval()
    y_pred = list()
    for step, batch in enumerate(tqdm(loader, desc="Inference iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
            # TODO: figure out why this exactly (it seems like it drops batches with only one sample)
            # NB: this routine is heavily inspired by:
            #     https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py
            # why don't we have the check on batch.batch as above?
        else:
            with torch.no_grad():
                pred = model(batch)
            y_pred.append(pred.detach().cpu())
    y_pred = torch.cat(y_pred, dim=0).numpy()
    return y_pred
            
def eval(model, device, loader, evaluator):
    """
        Evaluates a model over all the batches of a data loader.
    """
    
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Eval iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
            # TODO: figure out why this exactly (it seems like it drops batches with only one sample)
            # NB: this routine is heavily inspired by:
            #     https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py
            # why don't we have the check on batch.batch as above?
        else:
            with torch.no_grad():
                pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
    
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
        
    def _isomorphism(self, input_dict):
        p = input_dict.get('p', 2)
        eps = input_dict.get('eps', 0.01)
        preds = input_dict['y_pred']
        mm = torch.pdist(preds, p=p)
        correct = (mm < eps).sum().item()
        n = preds.shape[0]
        metric = correct / (n*(n-1))
        return metric
    
    def _accuracy(self, input_dict):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        metric = met.accuracy_score(y_true, y_pred)
        return metric
    
def load_dataset(name, root=os.path.join(ROOT_DIR, 'datasets')):
    # TODO: shall we move this to data.data_loading ?
    if name.startswith('sr'):
        dataset = SRDataset(os.path.join(root, 'SR_graphs'), name, 'isomorphism', 'isomorphism', exp_dim=5)
    else:
        raise NotImplementedError
    return dataset