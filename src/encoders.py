import collections

import numpy as np
from sklearn.preprocessing import StandardScaler


PAD, EOS, BOS, UNK = '<PAD>', '<EOS>', '<BOS>', "<UNK>"


class CategoricalEncoder:

    __type__ = 'categorical'
    
    def __init__(self, name, pad_token=PAD, eos_token=EOS, bos_token=BOS, unk_token=UNK,
                 fixed_vocab=False):
        self.index = collections.defaultdict()
        self.index.default_factory = lambda: len(self.index)
        self.pad_token, self.unk_token = pad_token, unk_token
        self.bos_token, self.eos_token = bos_token, eos_token
        self.reserved_tokens = list(filter(None, (pad_token, eos_token, bos_token, unk_token)))
        self.name = name
        self.fixed_vocab = fixed_vocab
        self._size = 0

        for token in self.reserved_tokens:
            self.index[token]

    def __getitem__(self, item):
        if self.fixed_vocab:
            return self.index.get(item, self.index[self.unk_token])
        return self.index[item]

    def __len__(self):
        return len(self.index)

    def size(self):
        return len(self)

    def fixate(self):
        self.fixed_vocab = True

    def to_dict(self):
        d = {
            'name': self.name,
            'type': 'categorical',
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'fixed_vocab': self.fixed_vocab,
        }
        d['index'] = dict(self.index)
        return d

    @staticmethod
    def from_dict(d):
        encoder = CategoricalEncoder(
            d['name'], pad_token=d['pad_token'], eos_token=d['eos_token'],
            bos_token=d['bos_token'], unk_token=d['unk_token'],
            fixed_vocab=d['fixed_vocab']
        )
        encoder.index.update(d['index'])
        return encoder

    @property
    def pad_index(self):
        return self.index[self.pad_token]

    @property
    def bos_index(self):
        return self.index[self.bos_token]

    @property
    def eos_index(self):
        return self.index[self.eos_token]

    @property
    def unk_index(self):
        return self.index[self.unk_token]

    def __repr__(self):
        return '<CategoricalEncoder({})>'.format(self.name)

    def transform(self, sample, feature_index):
        eos = [self.eos_index] if self.eos_token is not None else []
        bos = [self.bos_index] if self.bos_token is not None else []
        sample = bos + [self[elt] for elt in sample[feature_index]] + eos
        return sample


class ContinuousEncoder:

    __type__ = 'continuous'

    def __init__(self, name, bos=True, eos=True, scaler=StandardScaler):
        self.name = name
        self.bos, self.eos = bos, eos
        self.scaler = scaler()

    def fit(self, X):
        self.scaler.fit(np.array(X).reshape(-1, 1))
        return self

    def to_dict(self):
        params = {'name': self.name, 'type': 'continuous'}
        params.update(self.scaler.get_params())
        params['mean_'] = list(self.scaler.mean_)
        params['scale_'] = list(self.scaler.scale_)
        params['var_'] = list(self.scaler.var_)
        return params

    @staticmethod
    def from_dict(d):
        scaler = StandardScaler(
            copy=d['copy'], with_mean=d['with_mean'], with_std=d['with_std']
        )
        scaler.mean_ = np.array(d['mean_'])
        scaler.scale_ = np.array(d['scale_'])
        scaler.var_ = np.array(d['var_'])
        encoder = ContinuousEncoder(d['name'], d['bos'], d['eos'], scaler=scaler)
        return encoder

    def transform(self, sample, feature_index):
        bos = [0] if self.bos else []
        eos = [0] if self.eos else []
        sample = [float(elt) for elt in sample[feature_index]]
        sample = self.scaler.transform(np.array(sample).reshape(-1, 1))
        sample = sample.reshape(1, -1)[0].tolist()
        return bos + sample + eos

    def __repr__(self):
        return '<ContinuousEncoder({})>'.format(self.name)
