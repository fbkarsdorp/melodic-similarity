import time

import numpy as np
import torch

from livelossplot import PlotLosses
from metrics import evaluate_ranking


class EarlyStopException(Exception):
    def __init__(self, best, best_params, fails, val_scores):
        self.best = best
        self.best_params = best_params
        self.fails = fails
        self.val_scores = val_scores


def to_cpu(state_dict):
    for k, v in state_dict.items():
        state_dict[k] = v.to('cpu')
    return state_dict


def get_lr(optimizer):
    return next(param_group['lr'] for param_group in optimizer.param_groups)


def fit_model(train_loader, val_loader, model, optimizer, scheduler, 
              n_epochs, log_interval, plot=True, burnin=-1,
              patience=3, early_stop_score='MAP', eval_metric='cosine'):
    early_stop = {}
    early_stop['best'] = -float('inf')
    early_stop['best_params'] = to_cpu(model.state_dict())
    early_stop['fails'] = 0

    if plot:
        liveloss = PlotLosses()
    for epoch in range(n_epochs):
        logs = {}
        start_time = time.time()

        # Training
        train_loss = train_epoch(train_loader, model, optimizer)
        train_scores = {}
        # Turned off for optimize
        # if epoch > 0 and epoch % log_interval == 0:
            # train_scores = evaluate_ranking(model, train_loader, metric=eval_metric)

        elapsed = time.time() - start_time
        message = '\n' + '=' * 80
        message += '\nTrain:     '
        message += f' epoch: {epoch:2d}, time: {int(elapsed):d}s., loss: {train_loss:5.3f}'
        if 'silhouette' in train_scores:
            message += f', silouhette: {train_scores["silhouette"]:.2f}'
        message += '\n'

        # Validation
        start_time = time.time()
        val_loss = test_epoch(val_loader, model)
        val_scores = {}

        if epoch > 0 and epoch % log_interval == 0:
            train_label_set = list(set(train_loader.dataset.labels))
            val_scores = evaluate_ranking(
                model, val_loader, train_label_set, metric=eval_metric)

            # early stopping
            if val_scores[early_stop_score] > early_stop['best']:
                early_stop['best'] = val_scores[early_stop_score]
                early_stop['best_params'] = to_cpu(model.state_dict())
                early_stop['fails'] = 0
                early_stop['val_scores'] = val_scores
            else:
                early_stop['fails'] += 1
            if early_stop['fails'] >= patience:
                raise EarlyStopException(
                    early_stop['best'], early_stop['best_params'], early_stop['fails'],
                    early_stop['val_scores'])

        elapsed = time.time() - start_time

        message += 'Validation:'
        message += f' epoch: {epoch:2d}, time: {int(elapsed):d}s., loss: {val_loss:5.3f}'
        if 'silhouette' in val_scores:
            message += f', silhouette: {val_scores["silhouette"]:.2f}'
            message += f'\n            MAP: {val_scores["MAP"]:.2f}'
            message += f', MAP (seen): {val_scores["MAP seen labels"]:.2f}'
            message += f', MAP (unseen): {val_scores["MAP unseen labels"]:.2f}'
        message += '\n'
        message += '=' * 80 + '\n'
        print(message)

        logs['loss'] = train_loss
        logs['val_loss'] = val_loss
        for score, value in train_scores.items():
            logs[score] = value
        for score, value in val_scores.items():
            logs[f'val_{score}'] = value

        if epoch > burnin:
            scheduler.step(val_loss)

        if plot:
            liveloss.update(logs)
            liveloss.draw()

    # return data in case it never early stopped
    return early_stop


def train_epoch(loader, model, optimizer, max_norm=5.0):
    model.train()
    epoch_loss = 0

    for i, (data, target) in enumerate(loader):

        data = tuple((tuple(x.to(model.device()) if x is not None else x for x in d), l)
                     for d, l in data)
        if len(target) > 0:
            target = target.to(model.device())

        optimizer.zero_grad()
        outputs = model(*data)

        outputs = (outputs,) if not isinstance(outputs, tuple) else outputs
        loss_inputs = outputs
        if len(target) > 0:  # pairs have target labels; triplets don't
            loss_inputs += (target,)

        loss = model.objective.loss(*loss_inputs)
        epoch_loss += loss.item()
        loss.backward()
        # add clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if i % 5 == 0:
            curr_loss = epoch_loss / (i + 1)
            if hasattr(loader, 'batch_size') and loader.batch_size is not None:
                done = i * loader.batch_size / len(loader.dataset)
            else:
                done = i * len(outputs[0]) / len(loader.dataset)
            lr = get_lr(optimizer)
            message = f'Train {done:.0%}, loss {curr_loss:5.3f}, lr: {lr:.4f}'
            print(message, end='\r')

    return epoch_loss / (i + 1)


def test_epoch(loader, model):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for i, (data, target) in enumerate(loader):

            data = tuple((tuple(x.to(model.device()) if x is not None else x for x in d), l)
                         for d, l in data)
            if len(target) > 0:
                target = target.to(model.device())

            outputs = model(*data)

            outputs = (outputs,) if not isinstance(outputs, tuple) else outputs
            loss_inputs = outputs
            if len(target) > 0:
                loss_inputs += (target,)

            loss = model.objective.loss(*loss_inputs)
            val_loss += loss.item()

        return val_loss / (i + 1)
