import itertools
import os
import pickle
import shutil
import uuid

import altair as alt
import numpy as np
import pandas as pd
import scipy.special
import torch

from metrics import encode_sequences


def save_checkpoint(state, is_best, dirname='./', filename='checkpoint.pth.tar'):
    filename = os.path.join(dirname, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))


def save_encodings(data, model, fpath, suffix):
    with torch.no_grad():
        model.eval()
        embeddings = encode_sequences(data, model)
        results = []
        for i, embedding in enumerate(embeddings):
            results.append({
                'tunefamily': data.labels[i],
                'id': data.ids[i],
                'embedding': embedding
            })
        rid = str(uuid.uuid1())[:8]
        with open(f'{fpath}_{rid}_{suffix}.pkl', 'wb') as out:
            pickle.dump(results, out)


def plot_embeddings(embeddings, reducer, labels=None, ids=None, train_or_test=None):
    Y = reducer.fit_transform(embeddings)
    df = pd.DataFrame(Y, columns=['x', 'y'])
    df['label'] = labels
    df['id'] = ids
    df['data'] = train_or_test
    return alt.Chart(df).mark_point().encode(
        x='x', y='y', color='label', tooltip=['label', 'id', 'data']
    ).interactive()


def comb_index(n, k):
    count = scipy.special.comb(n, k, exact=True)
    index = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)


class IdentityScaler:
    def fit(self, X):
        return self

    def transform(self, X):
        return X
