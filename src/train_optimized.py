import os
import json

from train import run_experiment, get_arguments, make_seed


def read_parameters(fname):
    rows = [json.loads(line) for line in open(fname)]
    rows.sort(key=lambda i: i['dev_score'], reverse=True)
    return rows


def run(config, args):
    args_dict = vars(args)
    for parameter, value in config['params'].items():
        if parameter not in ('train', 'dev', 'test', 'cuda', 'results_path'):
            args_dict[parameter] = value
    print(args)
    run_experiment(args, config['seed'])


if __name__ == '__main__':
    args = get_arguments()
    configs = read_parameters(args.parameter_config)
    for config in configs[:args.n_best_parameters]:
        print(f'Running with dev score {config["dev_score"]:.2f}')
        run(config, args)
