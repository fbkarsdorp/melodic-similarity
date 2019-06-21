import argparse
import gzip
import json
import os
import random

from collections import defaultdict, Counter
from itertools import filterfalse
from itertools import groupby

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

import utils

# Example use:

# This selects songs from ANN background corpus, with year > 1850 and type 'vocal'
# and divides those in train and test set making sure the split is at the level of tunefamily

# dl = MTCDataLoader('mtcfsinst_sequences.jsonl')
# seqs = dl.applyFilters(
#    [
#        {'o_filter':'ann_bgcorpus'},
#        {'o_filter':('afteryear',1850)},
#        {'o_filter':'vocal'},
#        {'o_filter':'labeled'}
#    ]
# )
# train, test = dl.train_test_split(groupby='tunefamily', seq_iter=seqs)


def make_split(seqs, test_size=0.25, cross_class_size=0.25):
    labels = [seq["tunefamily"] for seq in seqs]
    labelset = set(labels)
    assert min(Counter(labels).values()) > 1
    index = {
        target: [i for i, label in enumerate(labels) if label == target]
        for target in labelset
    }
    for label in index:
        random.shuffle(index[label])
    pair_dict = {label: [] for label in labelset}
    for label, indexes in index.items():
        for i in range(0, len(indexes), 2):
            if i + 1 == len(indexes):
                assert len(indexes) % 2 != 0
                pair_dict[label][-1].append(indexes[i])
            else:
                a, b = indexes[i], indexes[i + 1]
                pair_dict[label].append([a, b])
    train, test = [], []
    n_unseen_test = int((1 - cross_class_size) * (test_size * len(seqs)))
    while n_unseen_test > 0:
        candidate = False
        label = np.random.choice(
            [label for label in pair_dict if len(pair_dict[label]) <= 5])
        for pair in pair_dict[label]:
            test.extend(pair)
            n_unseen_test -= len(pair)
        pair_dict = {y: pairs for y, pairs in pair_dict.items() if y != label}
    n_cross_test = int(cross_class_size * (test_size * len(seqs)))
    while n_cross_test > 0:
        label = np.random.choice(
            [label for label in pair_dict if len(pair_dict[label]) > 1])
        pair = pair_dict[label].pop()
        test.extend(pair)
        n_cross_test -= len(pair)
        if not any(len(pairs) > 1 for pairs in pair_dict.values()):
            break
    for label, pairs in pair_dict.items():
        train.extend(sum(pairs, []))
    assert len(train) + len(test) == len(seqs)
    return [seqs[i] for i in train], [seqs[i] for i in test]


class DataLoader:
    def __init__(self, jsonpath):
        self.jsonpath = jsonpath
        self.filterBank = {}  # defaultdict(lambda : False)
        self.featureExtractors = {}  # defaultdict(lambda: 0)

    def sequences(self):
        if self.jsonpath.endswith("gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(self.jsonpath, "r") as f:
            for line in f:
                yield json.loads(line)

    @staticmethod
    def writeJSON(json_out_path, seq_iter):
        if json_out_path.endswith(".gz"):
            opener = gzip.open
            mode = "wt"
        else:
            opener = open
            mode = "w"
        with opener(json_out_path, mode) as f:
            for seq in seq_iter:
                seq_json = json.dumps(seq)
                f.write(seq_json + "\n")

    def getFeatureNames(self):
        seqs = self.sequences()
        seq = next(seqs)  # get the names from first sequence
        return seq["features"].keys()

    def applyFilter(self, o_filter, invert=False, seq_iter=None):
        if seq_iter is None:
            seq_iter = self.sequences()
        filterer = filter
        if invert:
            filterer = filterfalse
        if type(o_filter) == tuple:
            return filterer(self.filterBank[o_filter[0]](*o_filter[1:]), seq_iter)
        else:
            return filterer(self.filterBank[o_filter], seq_iter)

    def applyFilters(self, filter_list, seq_iter=None):
        if seq_iter is None:
            seq_iter = self.sequences()
        for filt in filter_list:
            seq_iter = self.applyFilter(**filt, seq_iter=seq_iter)
        return seq_iter

    def registerFilter(self, name, o_filter):
        self.filterBank[name] = o_filter

    def selectFeatures(self, featlist, seq_iter=None):
        if seq_iter is None:
            seq_iter = self.sequences()
        for seq in seq_iter:
            features = {k: v for k, v in seq["features"].items() if k in featlist}
            seq["features"] = features
            yield seq

    def extractFeature(self, name, func, feats, seq_iter=None):
        if seq_iter is None:
            seq_iter = self.sequences()
        for seq in seq_iter:
            featvects = [seq["features"][x] for x in feats]
            args = zip(*featvects)
            newfeat = [func(*local_args) for local_args in args]
            seq["features"][name] = newfeat
            yield seq

    def registerFeatureExtractor(self, name, func, feats):
        self.featureExtractors[name] = (func, feats)

    def applyFeatureExtractor(self, name, seq_iter=None):
        if seq_iter is None:
            seq_iter = self.sequences()
        func, feats = self.featureExtractors[name]
        return self.extractFeature(name, func, feats, seq_iter)

    def concatAllFeatures(self, name="concat", seq_iter=None):
        feats = self.getFeatureNames()
        return self.extractFeature(
            name,
            func=lambda *args: " ".join([str(a) for a in args]),
            feats=feats,
            seq_iter=seq_iter,
        )

    def train_test_split(
        self,
        groupby=None,
        test_size=0.25,
        cross_class_size=0.25,
        random_state=None,
        seq_iter=None,
    ):

        if seq_iter is None:
            seq_iter = self.sequences()
        seqs = list(seq_iter)  # Oops... memory
        if groupby is None:
            train, test = make_split(
                seqs, test_size=test_size, cross_class_size=cross_class_size
            )
        else:
            gkf = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            group = [seq[groupby] for seq in seqs]
            train_ixs, test_ixs = next(gkf.split(seqs, groups=group))
            train = [seqs[ix] for ix in train_ixs]
            test = [seqs[ix] for ix in test_ixs]
        return train, test

    # heavy on memory
    def minClassSizeFilter(self, classfeature, mininum=0, seq_iter=None):
        if seq_iter is None:
            seq_iter = self.sequences()
        mem = defaultdict(list)
        for seq in seq_iter:
            mem[seq[classfeature]].append(seq)
            if len(mem[seq[classfeature]]) == mininum:
                for s in mem[seq[classfeature]]:
                    yield s
            elif len(mem[seq[classfeature]]) > mininum:
                yield seq

    # heavy on memory
    def maxClassSizeFilter(self, classfeature, maximum=100, seq_iter=None):
        if seq_iter is None:
            seq_iter = self.sequences()
        seqs = sorted(list(seq_iter), key=lambda x: x[classfeature])  # Allas
        for _, gr in groupby(seqs, key=lambda x: x[classfeature]):
            group = list(gr)
            if len(group) <= maximum:
                for g in group:
                    yield g


class MTCDataLoader(DataLoader):
    def __init__(self, jsonpath):
        super(MTCDataLoader, self).__init__(jsonpath)
        self.addMTCFilters()
        self.addMTCFeatureExtractors()

    def addMTCFilters(self):
        self.registerFilter("vocal", lambda x: x["type"] == "vocal")
        self.registerFilter("instrumental", lambda x: x["type"] == "instrumental")
        self.registerFilter("ann_bgcorpus", lambda x: x["ann_bgcorpus"] == True)
        self.registerFilter("freemeter", lambda x: x["freemeter"] == True)
        self.registerFilter("firstvoice", lambda x: x["id"][10:12] == "01")
        self.registerFilter("afteryear", lambda y: lambda x: x["year"] > y)
        self.registerFilter("beforeyear", lambda y: lambda x: x["year"] < y)
        self.registerFilter(
            "betweenyears", lambda l, h: lambda x: x["year"] > l and x["year"] < h
        )
        self.registerFilter("labeled", lambda x: x["tunefamily"] != "")
        self.registerFilter("unlabeled", lambda x: x["tunefamily"] == "")

        def inOGL(nlbid):
            rn = int(nlbid[3:9])
            return (rn >= 70000 and rn < 80250) or rn == 176897

        self.registerFilter("inOGL", lambda x: inOGL(x["id"]))
        self.registerFilter("inNLBIDs", lambda id_list: lambda x: x["id"] in id_list)
        self.registerFilter(
            "inTuneFamilies", lambda tf_list: lambda x: x["tunefamily"] in tf_list
        )
        inst_test_list = [
            "10373_0",
            "230_0",
            "4560_0",
            "5559_0",
            "3680_0",
            "1079_0",
            "288_1",
            "10075_0",
            "10121_0",
            "4652_0",
            "7016_0",
            "5542_0",
            "1324_0",
            "2566_0",
            "5448_1",
            "5389_0",
            "4756_0",
            "5293_0",
            "9353_0",
            "240_0",
            "5315_0",
            "7918_0",
            "5855_0",
            "5521_0",
            "7116_0",
            "371_0",
        ]
        self.registerFilter("inInstTest", lambda x: x["tunefamily"] in inst_test_list)

    def addMTCFeatureExtractors(self):
        self.registerFeatureExtractor(
            "full_beat_str",
            lambda x, y: str(x) + " " + str(y),
            ["beat_str", "beat_fraction_str"],
        )


class DataConf:
    def __init__(self, datadir="../data/", random_state=None):
        self.datadir = datadir
        self.random_state = random_state

    def getData(self, name, dev_size, test_size, cross_class_size):
        return getattr(self, name)(dev_size, test_size, cross_class_size)

    def cFS(self, dev_size, test_size, cross_class_size):
        """Use cFS for train/dev and testing.
        
        train/dev : cFS
        test: cFS
        
        Classes are shared between train/dev and test
        """

        ann_dl = MTCDataLoader(os.path.join(self.datadir, "mtcann_sequences.jsonl.gz"))
        train, test = ann_dl.train_test_split(
            test_size=test_size, cross_class_size=cross_class_size,
            random_state=self.random_state
        )
        train, dev = ann_dl.train_test_split(
            test_size=dev_size, cross_class_size=cross_class_size,
            seq_iter=train, random_state=self.random_state
        )
        return train, dev, test

    def test_cFS(self, dev_size, test_size, cross_class_size):
        """Use cFS for testing and unrelated melodies from MTC-FS-INST for train/dev.
        
        test: cFS: 360 melodies, 26 tune families.
        train/dev: unrelated melodies in MTC-FS-INST: vocal, year>1850, ann_bgcorpus==True
        
        Classes are not shared between train/test
        Classes are not shared between train/dev
        """

        fsinst_dl = MTCDataLoader(os.path.join(self.datadir, "mtcfsinst_sequences.jsonl.gz"))
        ann_dl = MTCDataLoader(os.path.join(self.datadir, "mtcann_sequences.jsonl.gz"))

        vocalbg = fsinst_dl.minClassSizeFilter(
            "tunefamily", 2,
            seq_iter=fsinst_dl.applyFilter(
                ("afteryear", 1850),
                seq_iter=fsinst_dl.applyFilter(
                    "vocal",
                    seq_iter=fsinst_dl.applyFilter("ann_bgcorpus")
                )
            )
        )

        train, dev = fsinst_dl.train_test_split(
            test_size=dev_size,
            groupby="tunefamily",
            random_state=self.random_state,
            seq_iter=vocalbg,
        )
        test = list(ann_dl.sequences())
        return train, dev, test

    def test_cINST(self, dev_size, test_size, cross_class_size):
        """Use curated Instrumental Corpus (cINST) for testing and unrelated 
        melodies from MTC-FS-INST for train/dev.
        
        test: cINST: c.380 instrumental melodies in 26 tune families.
        train/dev: unrelated melodies in MTC-FS-INST: year<1850, instrumental, not 
        in Instrumental Corpus
        
        Tune families <2 members and unlabeled melodies are excluded from train set.
        
        Classes are not shared between train/test
        Classes are not shared between train/dev
        """

        fsinst_dl = MTCDataLoader(os.path.join(self.datadir, "mtcfsinst_sequences.jsonl.gz"))

        inst_test_iter = fsinst_dl.applyFilter(
            ("beforeyear", 1850),
            seq_iter=fsinst_dl.applyFilter(
                "instrumental",
                seq_iter=fsinst_dl.applyFilter("inInstTest")
            )
        )

        inst_train_iter = fsinst_dl.minClassSizeFilter(
            "tunefamily", 2,
            seq_iter=fsinst_dl.applyFilter(
                ("beforeyear", 1850),
                seq_iter=fsinst_dl.applyFilter(
                    "instrumental",
                    seq_iter=fsinst_dl.applyFilter(
                        "inInstTest",
                        invert=True,
                        seq_iter=fsinst_dl.applyFilter('labeled')
                    )
                )
            )
        )

        train, dev = fsinst_dl.train_test_split(
            test_size=dev_size,
            groupby="tunefamily",
            random_state=self.random_state,
            seq_iter=inst_train_iter,
        )
        test = list(inst_test_iter)
        return train, dev, test

    def FS(self, dev_size, test_size, cross_class_size):
        """Use vocal part of MTC-FS-INST for testing and train/dev
        
        vocal part: year>1850, vocal
        
        Classes are shared between train and test
        Classes are not shared between train and dev
        
        Tune families <2 members and unlabeled melodies are excluded.
        """

        fsinst_dl = MTCDataLoader(
            os.path.join(self.datadir, "mtcfsinst_sequences.jsonl.gz")
        )

        selection = fsinst_dl.minClassSizeFilter(
            "tunefamily", 2,
            seq_iter=fsinst_dl.applyFilter(
                "vocal",
                seq_iter=fsinst_dl.applyFilter(
                    ("afteryear",1850),
                    seq_iter=fsinst_dl.applyFilter("labeled")
                )
            )
        )

        selection = list(selection)

        train, test = fsinst_dl.train_test_split(
            test_size=test_size,
            cross_class_size=cross_class_size,
            random_state=self.random_state,
            seq_iter=selection,
        )
        train, dev = fsinst_dl.train_test_split(
            test_size=dev_size,
            cross_class_size=cross_class_size,
            random_state=self.random_state,
            seq_iter=train,
        )
        return train, dev, test

    def INST(self, dev_size, test_size, cross_class_size):
        """Use instrumental part of MTC-FS-INST for testing and train/dev
        
        instrumental part: year<1850, instrumental
        
        Classes are shared between train and test
        Classes are not shared between train and dev
        
        Tune families <2 members and unlabeled melodies are excluded.
        """

        fsinst_dl = MTCDataLoader(
            os.path.join(self.datadir, "mtcfsinst_sequences.jsonl.gz")
        )

        selection = fsinst_dl.minClassSizeFilter(
            "tunefamily", 2,
            seq_iter=fsinst_dl.applyFilter(
                "instrumental",
                seq_iter=fsinst_dl.applyFilter(
                    ("beforeyear",1850),
                    seq_iter=fsinst_dl.applyFilter("labeled")
                )
            )
        )

        selection = list(selection)

        train, test = fsinst_dl.train_test_split(
            test_size=test_size,
            cross_class_size=cross_class_size,
            random_state=self.random_state,
            seq_iter=selection,
        )
        train, dev = fsinst_dl.train_test_split(
            test_size=dev_size,
            cross_class_size=cross_class_size,
            random_state=self.random_state,
            seq_iter=train,
        )
        return train, dev, test

    def FSINST(self, dev_size, test_size, cross_class_size):
        """Use full MTC-FS-INST for testing and train/dev
        
        Classes are shared between train and test
        Classes are not shared between train and dev
        
        Tune families <2 members and unlabeled melodies are excluded.
        """
        fsinst_dl = MTCDataLoader(
            os.path.join(self.datadir, "mtcfsinst_sequences.jsonl.gz")
        )

        selection = fsinst_dl.minClassSizeFilter(
            "tunefamily", 2, seq_iter=fsinst_dl.applyFilter("labeled")
        )

        selection = list(selection)

        train, test = fsinst_dl.train_test_split(
            test_size=test_size,
            cross_class_size=cross_class_size,
            random_state=self.random_state,
            seq_iter=selection,
        )
        train, dev = fsinst_dl.train_test_split(
            test_size=dev_size,
            cross_class_size=cross_class_size,
            random_state=self.random_state,
            seq_iter=train,
        )
        return train, dev, test


CONFIGS = "cFS", "test_cFS", "test_cINST", "FS", "INST", "FSINST"

def statistics(train, dev, test):
    train_labels = {s['tunefamily'] for s in train}
    dev_labels = {s['tunefamily'] for s in dev}
    test_labels = {s['tunefamily'] for s in test}
    if dev_labels:
        print('dev in train:', len(dev_labels.intersection(train_labels)) / len(dev_labels))
        print('dev size:', len(dev) / (len(dev) + len(train)))
    if test_labels:
        print('test in train:', len(test_labels.intersection(train_labels)) / len(test_labels))
        print('test size:', len(test) / (len(test) + len(dev) + len(train)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_size', type=float, default=0.25)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--cross_class_size', type=float, default=0.5)
    args = parser.parse_args()
    dl = MTCDataLoader("")  # to have one for writing
    if not os.path.exists('../data/datasets'):
        os.makedirs('../data/datasets')
    config = DataConf()
    for name in CONFIGS:
        print(name)
        fn = getattr(config, name)
        train, dev, test = fn(args.dev_size, args.test_size, args.cross_class_size)
        statistics(train, dev, test)
        dl.writeJSON(os.path.join("../data/datasets/", f"{name}_train.jsonl"), train)
        dl.writeJSON(os.path.join("../data/datasets/", f"{name}_dev.jsonl"), dev)
        dl.writeJSON(os.path.join("../data/datasets/", f"{name}_test.jsonl"), test)
