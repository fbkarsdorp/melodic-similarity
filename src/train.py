
import json
import argparse
import collections
import functools
import random
from datetime import datetime
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import models
import samplers
import datasets
import metrics
import utils

from dataloader import MTCDataLoader, DataConf, CONFIGS
from encoders import CategoricalEncoder, ContinuousEncoder
from trainer import fit_model, EarlyStopException
from utils import IdentityScaler


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)

# seed numpy, torch and random

def make_seed():
    now = datetime.now()
    seed = now.hour * 10000 + now.minute * 100 + now.second
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


def run_experiment(args, seed):
    if args.train is not None:
        train_dl = MTCDataLoader(args.train)
        if args.dev is None:
            train_data, dev_data = train_dl.train_test_split(test_size=args.devsize)
        else:
            dev_dl = MTCDataLoader(args.dev)
            train_data = list(train_dl.sequences())
            dev_data = list(dev_dl.sequences())
        if args.test is not None:
            test_dl = MTCDataLoader(args.test)
            test_data = list(test_dl.sequences())
        else:
            test_data = []
        print(f'Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}')

    elif args.dataconf:
        if args.dataconf not in CONFIGS:
            print(f"Error. {args.dataconf} is not a valid data configuration.", file=sys.stderr)
            print(f"Choose one of: {' '.join(DataConf.confs.keys())}", file=sys.stderr)
            raise SystemExit
        train_data, dev_data, test_data = DataConf().getData(
            args.dataconf, args.devsize, args.testsize, args.cross_class_size)
        if args.test:
            print("Warning. Command line argument --test_data ignored.", file=sys.stderr)

    if args.savetrain:
        MTCDataLoader.writeJSON(args.savetrain, train_data)
    if args.savedev:
        MTCDataLoader.writeJSON(args.savedev, dev_data)
    if args.savetest:
        MTCDataLoader.writeJSON(args.savetest, test_data)

    cat_encoders = [CategoricalEncoder(f) for f in args.categorical_features]
    scaler = (StandardScaler if args.scaler == 'zscore' else
              MinMaxScaler if args.scaler == 'minmax' else
              IdentityScaler)
    cont_encoders = [ContinuousEncoder(f, scaler=scaler) for f in args.continuous_features]
    encoders = cat_encoders + cont_encoders

    train_selector, dev_selector = None, None
    if args.precompute_examples:
        if args.example_type == 'pairs':
            train_selector = samplers.PairSelector(
                pos_neg_ratio=args.pn_ratio, random_sample=args.sample_ratio)
            dev_selector = samplers.PairSelector(
                pos_neg_ratio=args.pn_ratio)
        else:
            train_selector = samplers.TripletSelector(sample_ratio=args.sample_ratio)
            dev_selector = samplers.TripletSelector(sample_ratio=args.sample_ratio)

    dataset_constructor = (
        datasets.Dataset if args.online_sampler else
        datasets.DupletDataset if args.example_type == 'pairs' else
        datasets.TripletDataset)

    train = dataset_constructor(
        train_data, *encoders, batch_size=args.batch_size,
        selector=train_selector, label='tunefamily', train=True).fit()

    dev = dataset_constructor(
        dev_data, *encoders, batch_size=args.batch_size,
        selector=dev_selector, label='tunefamily', train=False).fit()

    if args.precompute_examples:
        print(train_selector)
        print(dev_selector)

    collate_fn = datasets.collate_fn

    if args.balanced_batch_sampler:
        train_batch_sampler = datasets.BalancedBatchSampler(
            train.labels, n_classes=args.n_classes, n_samples=args.n_samples)
        dev_batch_sampler = datasets.BalancedBatchSampler(
            dev.labels, n_classes=args.n_classes, n_samples=args.n_samples)
        train_loader = DataLoader(train, batch_sampler=train_batch_sampler,
                                  collate_fn=collate_fn, num_workers=args.n_workers)
        dev_loader = DataLoader(dev, batch_sampler=dev_batch_sampler,
                                collate_fn=collate_fn, num_workers=args.n_workers)
    elif not args.precompute_examples:
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=args.n_workers)
        dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate_fn,
                                num_workers=args.n_workers)
    else:
        train_loader, dev_loader = datasets.DataLoader(train), datasets.DataLoader(dev)

    device = 'cuda' if args.cuda else 'cpu'

    emb_dims = [(encoder.size(), args.emb_dim) for encoder in cat_encoders]
    if args.model.lower() == 'rnn':
        network = models.RNN(emb_dims, args.hid_dim, cont_features=len(cont_encoders),
                             n_layers=args.n_layers, cell=args.cell,
                             dropout=args.dropout, bidirectional=args.bidirectional)
    elif args.model.lower() == 'cnn':
        network = models.CNN(
            emb_dims, cont_features=len(cont_encoders),
            kernel_sizes=tuple(args.kernel_sizes), highway_layers=args.highway_layers,
            out_channels=args.out_channels, dropout=args.dropout)
    else:
        network = models.CNNRNN(
            emb_dims, cont_features=len(cont_encoders),
            kernel_sizes=tuple(args.kernel_sizes), highway_layers=args.highway_layers,
            out_channels=args.out_channels, dropout=args.dropout,
            cell=args.cell, bidirectional=args.bidirectional,
            n_layers=args.n_layers)

    if args.example_type == 'pairs':
        if not args.online_sampler:
            if args.loss == 'cosine':
                loss_fn = models.CosinePairLoss(weight=args.weight, margin=args.margin)
            else:
                loss_fn = models.EuclideanPairLoss(margin=args.margin)
            model = models.TwinNetwork(network, loss_fn).to(device) # margin 0.16, 0.4
        else:
            if args.loss == 'cosine':
                loss_fn = models.OnlineCosinePairLoss(
                    samplers.HardNegativePairSelector(), weight=args.weight,
                    margin=args.margin, cutoff=args.cutoff_cosine)
            else:
                loss_fn = models.OnlineEuclideanPairLoss(
                    samplers.HardNegativePairSelector(), margin=args.margin)
            model = models.Network(network, loss_fn).to(device)
    else:
        if not args.online_sampler:
            if args.loss == 'cosine':
                loss_fn = models.CosineTripletLoss(margin=args.margin)
            else:
                loss_fn = models.EuclidianTripletLoss(margin=args.margin)
            model = models.TripletNetwork(network, loss_fn).to(device)
        else:
            if args.loss == 'cosine':
                loss_fn = models.OnlineCosineTripletLoss(
                    samplers.NegativeTripletSelector(
                        method=args.negative_pair_selector, margin=args.margin),
                    margin=args.margin)
            else:
                loss_fn = models.OnlineEuclideanTripletLoss(
                    samplers.NegativeTripletSelector(
                        method=args.negative_pair_selector, margin=args.margin),
                    margin=args.margin)
            model = models.Network(network, loss_fn).to(device)

    print(model)

    for embedding in model.network.embs:
        embedding.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, patience=args.lr_scheduler, threshold=1e-4, cooldown=5)
    print(f'Number of parameters: {sum(p.nelement() for p in model.parameters())}')

    try:
        early_stop = fit_model(
            train_loader, dev_loader, model, optimizer, scheduler, args.epochs,
            args.log_interval, plot=False, patience=args.patience,
            early_stop_score=args.early_stop_score, eval_metric=args.loss)
        best_score = early_stop['best']
        fails = early_stop['fails']
        best_params = early_stop['best_params']
        val_scores = early_stop['val_scores']
    except EarlyStopException as e:
        print("Early stopping training")
        best_score = e.best
        fails = e.fails
        best_params = e.best_params
        val_scores = e.val_scores

    model.load_state_dict(best_params)
    model.eval()
    # serialize model if necessary

    print("Best", args.early_stop_score, best_score)

    if args.save_encodings is not None and args.dev is not None:
        utils.save_encodings(dev, model, args.save_encodings, 'dev')

    if args.test is not None:
        train_label_set = list(set(train_loader.dataset.labels))
        test = dataset_constructor(
            test_data, *encoders, batch_size=args.batch_size,
            label='tunefamily', train=False).fit()
        test_scores = metrics.evaluate_ranking(
            model, test, train_label_set=train_label_set, metric=args.loss)
        message = 'Testing:\n'
        message += f'  silouhette: {test_scores["silhouette"]:.3f}\n'
        message += f'  MAP: {test_scores["MAP"]:.3f}\n'
        message += f'  MAP (seen): {test_scores["MAP seen labels"]:.3f}\n'
        message += f'  MAP (unseen): {test_scores["MAP unseen labels"]:.3f}\n'
        message += f'  Margin: {test_scores["margin_score"]:.3f}'
        print(message)

    with open(f'{args.results_dir}/{args.results_path}', 'a+') as f:
        f.write(json.dumps({
            "params": vars(args),
            "dev_score": float(best_score),
            "val_scores": val_scores,
            "test_scores": test_scores if args.test is not None else {},
            "fails": fails,
            "seed": seed,
            "now": str(datetime.now())
        }, cls=NumpyEncoder) + '\n')

    if args.save_encodings is not None and args.test is not None:
        utils.save_encodings(test, model, args.save_encodings, 'test')
        
    return model


def get_arguments():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', type=str, default=None,
                       help='Path to train file.')
    group.add_argument('--dataconf', type=str, default=None,
                       help='Named data configuration.')
    parser.add_argument('--test', type=str, default=None,
                        help='Path to test file. Ignored if --dataconf is used.')
    parser.add_argument('--dev', type=str, default=None,
                        help='Path to dev file.')
    parser.add_argument('--savetrain', type=str, default=None,
                        help='Filename to save train dataset.')
    parser.add_argument('--savedev', type=str, default=None,
                        help='Filename to save dev dataset.')
    parser.add_argument('--savetest', type=str, default=None,
                        help='Filename to save test dataset.')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Random state for train/dev/test split.')
    parser.add_argument('--devsize', type=float, default=0.2,
                        help='Held-out proportion for validation.')
    parser.add_argument('--testsize', type=float, default=0.2,
                        help='Held-out proportion for testing.')    
    parser.add_argument('--cross_class_size', type=float, default=0.5,
                        help='Fraction of labels shared by test and train.')
    parser.add_argument('--categorical_features', type=str, nargs='*', default=[],
                        help='Categorical features to use.')
    parser.add_argument('--continuous_features', type=str, nargs='*', default=[],
                        help='Continuous features to use.')
    parser.add_argument('--scaler', type=str, choices=['zscore', 'minmax', 'none'],
                        default='none', help='Type of scaler to use for continuous features.')
    parser.add_argument('--concat_features', action='store_true',
                        help='Concat features into unique symbols.')
    parser.add_argument('--example_type', type=str, choices=['pairs', 'triplets'],
                        default='pairs')
    parser.add_argument('--pn_ratio', type=int, default=1,
                        help='Ratio of negative per positive pairs.')
    parser.add_argument('--sample_ratio', type=float, default=0.5,
                        help='Sample ratio of possible triplets.')
    parser.add_argument('--precompute_examples', action='store_true',
                        help='Precompute pairs or triplets.')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size (ignored with balanced batch sampler).')
    parser.add_argument('--balanced_batch_sampler', action='store_true',
                        help='Use balanced batch sampler.')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--online_sampler', action='store_true')
    parser.add_argument('--negative_pair_selector', type=str, default='hardest',
                        choices=['hardest', 'semihard', 'random'])
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes per batch in balanced batch sampler.')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples per class in balanced batch sampler.')
    parser.add_argument('--n_workers', type=int, default=5,
                        help='Number of data loader workers.')
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--cuda', action='store_true', help='Run on GPU.')
    parser.add_argument('--model', type=str, choices=['RNN', 'CNN', 'CNNRNN'],
                        default='RNN')
    parser.add_argument('--emb_dim', type=int, default=8,
                        help='Embedding dimension per feature.')
    parser.add_argument('--hid_dim', type=int, default=64,
                        help='Hidden dimension for RNN.')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Run RNN left to right and right to left.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of RNN layers.')
    parser.add_argument('--forget_bias', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout applied to feature embeddings during training.')
    parser.add_argument('--rnn_dropout', type=float, default=0, help='Dropout for RNN')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[5, 4, 3],
                        help='Kernel sizes used in CNN.')
    parser.add_argument('--out_channels', type=int, default=100,
                        help='Number of output channels for CNN.')
    parser.add_argument('--highway_layers', type=int, default=2, help='Number of highway layers (CNN)')
    parser.add_argument('--loss', type=str, choices=['cosine', 'euclidean'],
                        default='euclidean')
    parser.add_argument('--cutoff_cosine', action='store_true',
                        help='Use cutoff cosine loss.')
    parser.add_argument('--margin', type=float, default=1,
                        help='Margin used in Cosine and Euclidean Contrastive loss.')
    parser.add_argument('--weight', type=float, default=0.5,
                        help='Weight used in cosine loss.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lr_scheduler', type=int, default=1000,
                        help='Number of epochs to wait until validation loss improves.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=3,
                        help='Stop training after so many evaluations without improvement')
    parser.add_argument('--early_stop_score', default='MAP',
                        help='Score to use for early stopping')
    parser.add_argument('--results_path', default='results.jsonl',
                        help='File storing results one json per line for flexibility')
    parser.add_argument('--results_dir', default='../log',
                        help='Directory for storing early stopping results.')
    parser.add_argument('--tune_iterations', type=int, default=200,
                        help='Number of iterations for tuning parameters.')
    parser.add_argument('--parameter_config', type=str)
    parser.add_argument('--n_best_parameters', type=int, default=5)
    parser.add_argument('--save_encodings', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    run_experiment(args, make_seed())
