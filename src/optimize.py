import random
import numpy as np
import scipy.stats as stats

from train import get_arguments, run_experiment, make_seed


class truncnorm:
    def __init__(self, mu, std, lower=0, upper=1):
        a, b = (lower - mu) / std, (upper - mu) / std
        self.norm = stats.truncnorm(a, b, mu, std)

    def rvs(self):
        return float(self.norm.rvs())


class normint:
    def __init__(self, mu, std, lower, upper):
        self.norm = truncnorm(mu, std, lower, upper)

    def rvs(self):
        return int(round(self.norm.rvs())) // 2 * 2


class sample_kernel_sizes:
    def rvs(self):
        pop_sizes = np.array([1, 2, 3, 4, 5, 6, 7, 9, 12])
        n_kernels = random.randint(2, len(pop_sizes))  # at least 2
        sizes = np.sort(np.random.choice(pop_sizes, n_kernels, replace=False))
        return list(sizes)


class sample_features:
    def __init__(self, features):
        self.features = features

    def rvs(self):
        n = random.randint(1, len(self.features))
        return list(np.random.choice(self.features, size=n, replace=False))


cat_features = [
    'scaledegree', 'pitch', #'midipitch', 'pitch40'
    'metriccontour', 'imacontour'
]

cont_features = [
    'duration', 'beat', 'beatstrength', 'imaweight', 'phrasepos'
]

sampling_config = {
    "n_classes": [5, 10, 20],
    "n_samples": [5, 10, 20],
    "margin": truncnorm(0.5, 0.3, lower=0.1, upper=1),
    "weight": truncnorm(0.5, 0.3, lower=0.1, upper=0.9),
    "cutoff_cosine": [True, False],
    "lr": truncnorm(0.001, 0.001, 0.0001, 0.01),
    "example_type": ["pairs", "triplets"],
    "negative_pair_selector": ["semihard", "random"]
}


rnn_config = {
    "model": ["RNN"],
    "cell": ["GRU", "LSTM"],
    "emb_dim": normint(8, 3, lower=4, upper=16),
    "hid_dim": [64, 128, 256],
    "n_layers": [1, 2, 3],
    "bidirectional": [True, False],
    "dropout": truncnorm(0.5, 0.1),
    "forget_bias": [True, False],
    "rnn_dropout": truncnorm(0, 0.1, lower=0, upper=0.5)
}

cnn_config = {
    "model": ["CNN"],
    "emb_dim": normint(8, 3, lower=4, upper=16),
    "out_channels": [32, 64, 124],
    "kernel_sizes": sample_kernel_sizes(),
    "highway_layers": [0, 1, 2, 3],
    "n_classes": [5, 10, 20],
    "n_samples": [5, 10, 20],
    "dropout": truncnorm(0.5, 0.1),
}

cnn_rnn_config = dict(dict(cnn_config, **rnn_config), model=['CNNRNN'])
rnn_config = dict(rnn_config, **sampling_config)
cnn_config = dict(cnn_config, **sampling_config)
cnn_rnn_config = dict(cnn_rnn_config, **sampling_config)


def sample_config(config):
    return {
        k: random.choice(p) if isinstance(p, (tuple, list)) else p.rvs()
        for k, p in config.items()
    }


def run(config, args, n_iter=200):
    for i in range(n_iter):
        args_dict = vars(args)
        args_dict['categorical_features'] = cat_features
        args_dict['continuous_features'] = cont_features
        args_dict.update(sample_config(config))
        print(args)
        seed = make_seed()
        run_experiment(args, seed)


if __name__ == "__main__":
    args = get_arguments()
    if args.model == 'RNN':
        config = rnn_config
    elif args.model == 'CNN':
        config = cnn_config
    else:
        config = cnn_rnn_config
    run(config, args, n_iter=args.tune_iterations)
