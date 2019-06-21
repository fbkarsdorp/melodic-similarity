import collections
import csv
import functools
import itertools

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F

import utils


class PairSelector:
    def __init__(self, pos_neg_ratio=0, random_sample=1):
        self.pos_neg_ratio = pos_neg_ratio
        self.random_sample = random_sample

    def _get_pairs(self, labels, all=False):
        labels = np.array(labels)
        all_pairs = utils.comb_index(len(labels), 2)
        positive_pairs = all_pairs[
            (labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()
        ]
        positive_pairs = np.hstack(
            (positive_pairs, np.ones(len(positive_pairs), dtype=int)[:, np.newaxis]))
        negative_pairs = all_pairs[
            (labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()
        ]
        negative_pairs = np.hstack(
            (negative_pairs, np.zeros(len(negative_pairs), dtype=int)[:, np.newaxis]))
        if not all and self.pos_neg_ratio:
            n_samples = int(len(positive_pairs) * self.pos_neg_ratio)
            indexes = np.random.permutation(len(negative_pairs))[:n_samples]
            negative_pairs = negative_pairs[indexes]
        return positive_pairs, negative_pairs

    def get_examples(self, labels, *args):
        positive_pairs, negative_pairs = self._get_pairs(labels)
        positive_pairs = positive_pairs[:int(self.random_sample * len(positive_pairs))]
        negative_pairs = negative_pairs[:int(self.random_sample * len(negative_pairs))]
        self.n_positive_pairs, self.n_negative_pairs = len(positive_pairs), len(negative_pairs)
        pairs = positive_pairs.tolist() + negative_pairs.tolist()
        np.random.shuffle(pairs)
        return pairs

    def __repr__(self):
        ratio = self.n_positive_pairs / (self.n_positive_pairs + self.n_negative_pairs)
        s = f'{self.__class__.__name__}(positives: {self.n_positive_pairs}, '
        s += f'negatives: {self.n_negative_pairs}, Pos-neg ratio: {ratio:.2%})'
        return s


class AlignmentBasedPairSelector(PairSelector):
    def __init__(self, pos_neg_ratio=0, hard_example_ratio=0, alignment_path=None):
        super(AlignmentBasedPairSelector, self).__init__(pos_neg_ratio=pos_neg_ratio)
        self.alignment_scores = collections.defaultdict(lambda: {})
        self.hard_example_ratio = hard_example_ratio

        with open(alignment_path) as f:
            for source, target, score, y in csv.reader(f):
                if y == '0':  # we don't need true pairs
                    self.alignment_scores[source][target] = float(score)

    def get_examples(self, labels, ids):
        positive_pairs, negative_pairs = self._get_pairs(labels, all=True)
        ids = np.array(ids)
        negative_pairs = sorted(
            negative_pairs, key=lambda i: self.alignment_scores[ids[i[0]]][ids[i[1]]])
        if self.pos_neg_ratio:
            n_samples = int(len(positive_pairs) * self.pos_neg_ratio)
            if self.hard_example_ratio:
                n_hard_samples = int(self.hard_example_ratio * n_samples)
                hard_pairs = negative_pairs[:n_hard_samples]
                self.n_hard_negatives = len(hard_pairs)
                easier_pairs = negative_pairs[n_hard_samples:]
                np.random.shuffle(easier_pairs)
                random_pairs = easier_pairs[:n_samples - n_hard_samples]
                self.n_random_negatives = len(random_pairs)
                negative_pairs = hard_pairs + random_pairs
            else:
                negative_pairs = negative_pairs[:n_samples]
        self.n_positive_pairs, self.n_negative_pairs = len(positive_pairs), len(negative_pairs)
        pairs = positive_pairs.tolist() + negative_pairs
        np.random.shuffle(pairs)
        return pairs

    def __repr__(self):
        ratio = self.n_positive_pairs / (self.n_positive_pairs + self.n_negative_pairs)
        s = f'{self.__class__.__name__}(positives: {self.n_positive_pairs}, '
        s += f'negatives: {self.n_negative_pairs}'
        if hasattr(self, 'n_hard_negatives'):
            s += f' (hard: {self.n_hard_negatives}, random: {self.n_random_negatives})'
        s += f', Pos-neg ratio: {ratio:.2%})'
        return s


def condensed_index(i, j, n):
    return i * n + j - i * (i + 1) / 2 - i - 1


class HardNegativePairSelector:
    def __init__(self, cpu=True):
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        dm = torch.pdist(embeddings)
        labels = labels.cpu().data.numpy()
        all_pairs = torch.LongTensor(utils.comb_index(len(labels), 2))
        pos_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        neg_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        dists = dm[condensed_index(neg_pairs[:, 0], neg_pairs[:, 1], embeddings.shape[0])].cpu()
        # find most similar negatives
        _, top = torch.topk(dists, len(pos_pairs), largest=False)
        neg_pairs = neg_pairs[torch.LongTensor(top)]

        return pos_pairs, neg_pairs


def hardest_negative(dists):
    hardest = np.argmax(dists)
    idx = None
    if dists[hardest] > 0:
        idx = hardest
    return idx


def random_hard_negative(dists):
    hard_ones = np.where(dists > 0)[0]
    idx = None
    if len(hard_ones) > 0:
        idx = np.random.choice(hard_ones)
    else:
        idx = np.random.randint(0, dists.shape[0])
    return idx


def semihard_negative(dists, margin=1):
    semi_hard_ones = np.where((dists < margin) & (dists > 0))[0]
    idx = None
    if len(semi_hard_ones) > 0:
        idx = np.random.choice(semi_hard_ones)
    return idx


class NegativeTripletSelector:
    def __init__(self, method='hardest', margin=1, cpu=True):
        self.cpu = cpu
        if method == 'hardest':
            self.selection_fn = hardest_negative
        elif method == 'random':
            self.selection_fn = random_hard_negative
        elif method == 'semihard':
            self.selection_fn = functools.partial(semihard_negative, margin=margin)
        self.margin = 1

    def get_triplets(self, embs, labels, selection_fn=None):
        if self.cpu:
            embs = embs.cpu()
        dm = torch.pdist(embs).cpu()
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            mask = labels == label
            if mask.sum() < 2:
                continue  # no labels with only 1 positive
            label_idx = np.where(mask)[0]
            neg_idx = torch.LongTensor(np.where(np.logical_not(mask))[0])
            pos_pairs = torch.LongTensor(list(itertools.combinations(label_idx, 2)))
            pos_dists = dm[condensed_index(pos_pairs[:, 0], pos_pairs[:, 1], embs.shape[0])]
            for (i, j), dist in zip(pos_pairs, pos_dists):
                loss = dist - dm[condensed_index(i, neg_idx, embs.shape[0])] + self.margin
                loss = loss.data.cpu().numpy()
                if selection_fn is None:
                    hard_idx = self.selection_fn(loss)
                else:
                    hard_idx = selection_fn(loss)
                if hard_idx is not None:
                    triplets.append([i, j, neg_idx[hard_idx]])
        if not triplets:
            print('No triplets found... Sampling random hard ones.')
            triplets = self.get_triplets(embs, torch.LongTensor(labels), random_hard_negative)
        return torch.LongTensor(triplets)


class TripletSelector:

    def __init__(self, sample_ratio=0):
        self.sample_ratio = sample_ratio

    def _get_triplets(self, labels):
        labels = np.array(labels)
        triplets = []
        for label in set(list(labels)):
            label_mask = (labels == label)
            label_indexes = np.where(label_mask)[0]
            if len(label_indexes) < 2:
                continue
            negative_indexes = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(itertools.combinations(label_indexes, 2))
            triplets.extend([
                [anchor[0], anchor[1], negative_index]
                for anchor in anchor_positives for negative_index in negative_indexes
            ])
        return triplets

    def get_examples(self, labels, *args):
        triplets = self._get_triplets(labels)
        np.random.shuffle(triplets)
        if self.sample_ratio:
            n_samples = int(len(triplets) * self.sample_ratio)
            triplets = triplets[:n_samples]
        self.n_examples = len(triplets)
        return triplets

    def __repr__(self):
        return f'{self.__class__.__name__}(examples: {self.n_examples})'


class AlignmentBasedTripletSelector(TripletSelector):
    def __init__(self, alignment_path=None, sample_ratio=0, hard_example_ratio=0):
        super(AlignmentBasedTripletSelector, self).__init__(sample_ratio=sample_ratio)
        self.alignment_scores = collections.defaultdict(lambda: {})
        self.hard_example_ratio = hard_example_ratio

        with open(alignment_path) as f:
            for source, target, score, y in csv.reader(f):
                if y == '0':  # we don't need true pairs
                    self.alignment_scores[source][target] = float(score)

    def get_examples(self, labels, ids):
        triplets = self._get_triplets(labels)
        ids = np.array(ids)
        triplets.sort(key=lambda triplet: self.alignment_scores[ids[triplet[1]]][ids[triplet[2]]])
        if self.sample_ratio:
            n_samples = int(len(triplets) * self.sample_ratio)
            if self.hard_example_ratio:
                self.n_hard_examples = int(self.hard_example_ratio * n_samples)
                self.n_random_examples = n_samples - self.n_hard_examples
                hard_examples = triplets[:self.n_hard_examples]
                easier_examples = triplets[self.n_hard_examples:]
                np.random.shuffle(easier_examples)
                random_examples = easier_examples[:n_samples - self.n_hard_examples]
                triplets = hard_examples + random_examples
            else:
                triplets = triplets[:n_samples]
        self.n_examples = len(triplets)
        np.random.shuffle(triplets)
        return triplets

    def __repr__(self):
        s = f'{self.__class__.__name__}(examples: {self.n_examples}'
        if hasattr(self, 'n_hard_examples'):
            s += f', hard: {self.n_hard_examples}, random: {self.n_random_examples}'
        s += ')'
        return s
