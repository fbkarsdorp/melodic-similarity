import collections
import functools
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler

from encoders import CategoricalEncoder, ContinuousEncoder


class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = dataset.batch_size

    def __iter__(self):
        for elt in self.dataset:
            yield elt

    def __len__(self):
        return len(self.dataset)


class Dataset:
    def __init__(self, data, *encoders, max_seq_len=None,
                 batch_size=1, selector=None, label='label', train=True):

        self.sequences = [
            [entry['features'][encoder.name][:max_seq_len] for encoder in encoders]
            for entry in data
        ]

        self.labels = [entry[label] for entry in data]
        self.labelset = set(self.labels)
        self.label_encoder = CategoricalEncoder('label')
        self.ids = [entry['id'] for entry in data]
        
        self.label2indices = {
            target: [i for i, label in enumerate(self.labels) if label == target]
            for target in self.labelset
        }

        self.encoders = encoders
        self.train = train
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
        if selector is not None:
            self.dataset = selector.get_examples(self.labels, self.ids)
        else:
            self.dataset = []

    def fit(self):
        cont_X = collections.defaultdict(list)
        for sequence in self.sequences:
            for i, encoder in enumerate(self.encoders):
                if encoder.__type__ == 'categorical':
                    for item in set(sequence[i]):
                        encoder[item]  # index the item
                else:
                    for item in sequence[i]:
                        cont_X[i].append(float(item))
        for i, X in cont_X.items():
            self.encoders[i].fit(X)
        for encoder in self.encoders:
            if encoder.__type__ == 'categorical':
                encoder.fixate()
        return self

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        x0, y = self.sequences[index], self.labels[index]
        x0 = self.transform(x0)
        return (x0, None), self.label_encoder[y]

    def __len__(self):
        return len(self.sequences)

    def transform(self, x):
        # categorical features * L, continuous features * L
        cat_X = torch.LongTensor(
            [encoder.transform(x, i) for i, encoder in enumerate(self.encoders)
             if encoder.__type__ == 'categorical'])
        cont_X = torch.FloatTensor(
            [encoder.transform(x, i) for i, encoder in enumerate(self.encoders)
             if encoder.__type__ == 'continuous'])
        return cat_X, cont_X


class DupletDataset(Dataset):
    def __init__(self, data, *encoders, max_seq_len=None,
                 batch_size=1, selector=None, label='label', train=True):
        super(DupletDataset, self).__init__(
            data, *encoders, max_seq_len=max_seq_len, batch_size=batch_size,
            selector=selector, label=label, train=train)

        if not self.dataset and not self.train:
            rnd = np.random.RandomState(101)
            positive_pairs = [
                (i, rnd.choice(self.label2indices[self.labels[i]]), 1)
                for i in range(0, len(self.sequences), 2)
            ]
            negative_pairs = [
                (i, rnd.choice(self.label2indices[np.random.choice(
                    list(self.labelset - {self.labels[i]}))]),
                 0)
                for i in range(1, len(self.sequences), 2)
            ]
            self.test_pairs = positive_pairs + negative_pairs

    def __iter__(self):
        np.random.shuffle(self.dataset)
        pairs = []
        for i in range(len(self.dataset)):
            pairs.append(self[i])
            if len(pairs) == self.batch_size:
                yield collate_fn(pairs)
                pairs = []
        if pairs:
            yield collate_fn(pairs)

    def __getitem__(self, index):
        if self.dataset:  # precomputed pairs
            i, j, y = self.dataset[index]
            x0, x1 = self.sequences[i], self.sequences[j]
        elif self.train:  # random pairs
            y = np.random.randint(0, 2)
            x0, label0 = self.sequences[index], self.labels[index]
            if len(self.label2indices[label0]) > 1 and y == 1:  # looking for a true pair
                twin_index = index
                while twin_index == index:
                    twin_index = np.random.choice(self.label2indices[label0])
            else:
                twin_label = np.random.choice(list(self.labelset - {label0}))
                twin_index = np.random.choice(self.label2indices[twin_label])
            x1 = self.sequences[twin_index]
        else:  # preset test pairs
            x0 = self.sequences[self.test_pairs[index][0]]
            x1 = self.sequences[self.test_pairs[index][1]]
            y = self.test_pairs[index][2]
        x0, x1 = self.transform(x0), self.transform(x1)
        return (x0, x1), y

    def __len__(self):
        return len(self.dataset) if self.dataset else len(self.sequences)


class TripletDataset(Dataset):
    def __init__(self, data, *encoders, max_seq_len=None,
                 batch_size=1, selector=None, label='label', train=True):
        super(TripletDataset, self).__init__(
            data, *encoders, batch_size=batch_size, selector=selector,
            label=label, train=train)

        if not self.dataset and not self.train:
            rnd = np.random.RandomState(101)
            triplets = [[
                i,
                rnd.choice(self.label2indices[self.labels[i]]),
                rnd.choice(self.label2indices[np.random.choice(
                    list(self.labelset - {self.labels[i]}))])
            ] for i in range(len(self.sequences))]
            self.test_triplets = triplets

    def __iter__(self):
        np.random.shuffle(self.dataset)
        triplets = []
        for i, j, k in self.dataset:
            x0, x1, x2 = self.sequences[i], self.sequences[j], self.sequences[k]
            x0 = self.transform(x0)
            x1 = self.transform(x1)
            x2 = self.transform(x2)
            triplets.append(((x0, x1, x2), None))
            if len(triplets) == self.batch_size:
                yield collate_fn(triplets)
                triplets = []
        if triplets:
            yield collate_fn(triplets)

    def __getitem__(self, index):
        if self.dataset:
            i, j, k = self.dataset[index]
            x0, x1, x2 = self.sequences[i], self.sequences[j], self.sequences[k]
            x0 = self.transform(x0)
            x1 = self.transform(x1)
            x2 = self.transform(x2)
            return (x0, x1, x2), None
        if self.train:
            x0, y = self.sequences[index], self.labels[index]
            twin_index = index
            while twin_index == index:
                twin_index = np.random.choice(self.label2indices[y])
            x1 = self.sequences[twin_index]
            y2 = np.random.choice(list(self.labelset - {y}))
            x2 = self.sequences[np.random.choice(self.label2indices[y2])]
            x0 = self.transform(x0)
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        else:
            i, j, k = self.test_triplets[index]
            x0, x1, x2 = self.sequences[i], self.sequences[j], self.sequences[k]
            x0 = self.transform(x0)
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        return (x0, x1, x2), None

    def __len__(self):
        return len(self.dataset) if self.dataset else len(self.sequences)


def collate_fn(data):
    data, labels = zip(*data)

    def pad(seqs):
        cats, conts = zip(*seqs)
        if len(cats[0]):
            lengths = [len(c[0]) for c in cats]
        else: 
            lengths = [len(c[0]) for c in conts]
        lengths = torch.LongTensor(lengths)
        _, sort = torch.sort(lengths, descending=True)
        _, unsort = sort.sort()
        length = lengths.max().item()
        cat_seqs, cont_seqs = [], []
        for cat, cont in seqs:
            if len(cat):
                cat_seqs.append(F.pad(cat, (0, length - cat.size(1))))
            if len(cont):
                cont_seqs.append(F.pad(cont, (0, length - cont.size(1))))
        cat_seqs = torch.stack(cat_seqs)[sort] if cat_seqs else None
        cont_seqs = torch.stack(cont_seqs)[sort] if cont_seqs else None
        return (cat_seqs, cont_seqs), (lengths[sort], unsort)

    if data[0][1] is None:
        data, _ = zip(*data)
        return (pad(data),), torch.FloatTensor(labels)

    if len(data[0]) == 2:  # duplet data
        positives, negatives = zip(*data)
        return (pad(positives), pad(negatives)), torch.FloatTensor(labels)

    elif len(data[0]) == 3:  # triplet data
        anchors, positives, negatives = zip(*data)
        return (pad(anchors), pad(positives), pad(negatives)), []

    else:
        raise ValueError("Data format not understood.")


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labelset = list(set(labels))
        self.index = {
            target: [i for i, label in enumerate(self.labels) if label == target]
            for target in self.labelset
        }
        self.shuffle_label_indices()

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(labels)
        self.batch_size = n_samples * n_classes

    def shuffle_label_indices(self):
        for label in self.labelset:
            np.random.shuffle(self.index[label])

    def __iter__(self):
        count = 0
        self.shuffle_label_indices()
        while count < self.n_dataset:
            idx = {label: 0 for label in self.labelset}
            np.random.shuffle(self.labelset)
            indices, i = [], 0
            while len(indices) < self.batch_size:
                c = self.labelset[i]
                j = idx[c]
                if j < len(self.index[c]):
                    samples = self.index[c][j: j + self.n_samples]
                    samples = samples[:self.batch_size - len(indices)]
                    indices.extend(samples)
                    idx[c] += len(samples)
                i += 1
                if i == len(self.labelset):
                    i = 0  # do another pass
            yield indices
            count += len(indices)

    def __len__(self):
        return self.n_dataset
