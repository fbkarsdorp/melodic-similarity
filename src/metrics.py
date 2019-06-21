import numpy as np
import torch

import sklearn.metrics as metrics


def encode_sequences(dataset, model):
    embs = []
    for i, seq in enumerate(dataset.sequences):
        x0 = tuple(d.unsqueeze(0).to(model.device()) if len(d) else None
                   for d in dataset.transform(seq))
        length = [x0[0].size(-1) if x0[0] is not None else x0[1].size(-1)]
        length = torch.LongTensor(length)
        embs.append(model.get_embedding(x0, (length, [0])).cpu().numpy()[0])
    return np.array(embs)


def mean_average_precision(embs, labels, metric):
    sim_matrix = 1 - metrics.pairwise_distances(embs, metric=metric)
    scores = []
    for i, sims in enumerate(sim_matrix):
        mask = np.arange(sims.shape[0]) != i # filter query
        query_y = labels[i]
        target_y = (labels[mask] == query_y).astype(int)
        if target_y.sum() > 0:
            scores.append(metrics.average_precision_score(target_y, sims[mask]))
    return np.mean(scores)


def subset_average_precision(embs, mask, labels, metric):
    sim_matrix = 1 - metrics.pairwise_distances(embs[mask], embs, metric=metric)
    scores = []
    for i, sims in zip(np.where(mask)[0], sim_matrix):
        m = np.arange(embs.shape[0]) != i
        query_y = labels[i]
        target_y = (labels[m] == query_y).astype(int)
        if target_y.sum() > 0:
            scores.append(metrics.average_precision_score(target_y, sims[m]))
    return np.mean(scores)


def margin_score(embs, labels, metric):
    sim_matrix = 1 - metrics.pairwise_distances(embs, metric=metric)
    scores = []
    for i, sims in enumerate(sim_matrix):
        mask = np.arange(sims.shape[0]) != i # filter query
        query_y = labels[i]
        target_y = (labels[mask] == query_y).astype(int)
        if target_y.sum() > 0:
            sort = np.argsort(sims[mask])[::-1]
            highest_irrelevant = np.argmin(target_y[sort])
            lowest_relevant = np.where(target_y[sort])[0][-1]
            scores.append(abs(highest_irrelevant - lowest_relevant))
    return np.mean(scores)


def evaluate_ranking(model, query_loader, train_label_set=None, metric='cosine'):
    scores = {}

    with torch.no_grad():
        model.eval()

        if isinstance(query_loader, torch.utils.data.dataloader.DataLoader):
            qembs = encode_sequences(query_loader.dataset, model)
            qlabels = np.array(query_loader.dataset.labels)
        else:
            qembs = encode_sequences(query_loader, model)
            qlabels = np.array(query_loader.labels)

        # compute silhouette score
        scores['silhouette'] = metrics.silhouette_score(qembs, qlabels, metric=metric)
        scores['margin_score'] = margin_score(qembs, qlabels, metric)

        if train_label_set is not None:
            # compute mean average precision
            scores['MAP'] = mean_average_precision(qembs, qlabels, metric)

            # compute MAP for seen classes
            mask = np.isin(qlabels, list(train_label_set))
            if mask.sum() == 0:
                scores['MAP seen labels'] = 0
            else:
                scores['MAP seen labels'] = subset_average_precision(
                    qembs, mask, qlabels, metric)

            # compute MAP for unseen classes
            if mask.sum() == len(qlabels):
                scores['MAP unseen labels'] = 0
            else:
                scores['MAP unseen labels'] = subset_average_precision(
                    qembs, ~mask, qlabels, metric)

    return scores
