import json
from collections import Counter
import numpy as np
from dataloader import MTCDataLoader

datasets = ['FSINST_dev.jsonl','FSINST_test.jsonl','FSINST_train.jsonl','FS_dev.jsonl','FS_test.jsonl','FS_train.jsonl','INST_dev.jsonl','INST_test.jsonl','INST_train.jsonl','cFS_dev.jsonl','cFS_test.jsonl','cFS_train.jsonl','test_cFS_dev.jsonl','test_cFS_test.jsonl','test_cFS_train.jsonl','test_cINST_dev.jsonl','test_cINST_test.jsonl','test_cINST_train.jsonl']
datanames = ['cFS', 'test_cFS', 'test_cINST', 'INST', 'FS', 'FSINST']

for name in datanames:
    with open('../data/datasets/'+name+'_train.jsonl','r') as f:
        trainseqs=[json.loads(line) for line in f]
    with open('../data/datasets/'+name+'_dev.jsonl','r') as f:
        devseqs=[json.loads(line) for line in f]
    with open('../data/datasets/'+name+'_test.jsonl','r') as f:
        testseqs=[json.loads(line) for line in f]

    fullset = trainseqs + devseqs + testseqs

    traintfs=[seq['tunefamily'] for seq in trainseqs]
    devtfs=[seq['tunefamily'] for seq in devseqs]
    testtfs=[seq['tunefamily'] for seq in testseqs]
    fulltfs=[seq['tunefamily'] for seq in fullset]

    trainy=[seq['year'] for seq in trainseqs]
    devy=[seq['year'] for seq in devseqs]
    testy=[seq['year'] for seq in testseqs]
    fully=[seq['year'] for seq in fullset]

    traint=[seq['type'] for seq in trainseqs]
    devt=[seq['type'] for seq in devseqs]
    testt=[seq['type'] for seq in testseqs]
    fullt=[seq['type'] for seq in fullset]

    c_train = Counter(traintfs)
    c_dev = Counter(devtfs)
    c_test = Counter(testtfs)
    c_full = Counter(fulltfs)

    if True:
        #print(name)
        print( len(fullset), end=' ' )
        print( len(set(fulltfs)), end=' ' )
        print( len(set(traintfs).intersection(set(fulltfs))), end=' ')
        print( np.mean(list(c_full.values())), end=' ')
        print( np.std(list(c_full.values())), end=' ')
        print( np.median(list(c_full.values())), end=' ')
        print( np.min(list(c_full.values())), end = ' ')
        print( np.max(list(c_full.values())), end = ' ')
        print( np.min(fully), end = ' ')
        print( np.max(fully), end = ' ')
        print( ','.join(list(set(fullt))))

    if True:
        print( len(trainseqs), end=' ' )
        print( len(set(traintfs)), end=' ' )
        print( len(set(traintfs).intersection(set(traintfs))), end=' ')
        print( np.mean(list(c_train.values())), end=' ')
        print( np.std(list(c_train.values())), end=' ')
        print( np.median(list(c_train.values())), end=' ')
        print( np.min(list(c_train.values())), end = ' ')
        print( np.max(list(c_train.values())), end = ' ')
        print( np.min(trainy), end = ' ')
        print( np.max(trainy), end = ' ')
        print( ','.join(list(set(traint))))

        print( len(devseqs), end=' ' )
        print( len(set(devtfs)), end=' ' )
        print( len(set(traintfs).intersection(set(devtfs))), end=' ')
        print( np.mean(list(c_dev.values())), end=' ')
        print( np.std(list(c_dev.values())), end=' ')
        print( np.median(list(c_dev.values())), end=' ')
        print( np.min(list(c_dev.values())), end = ' ')
        print( np.max(list(c_dev.values())), end = ' ')
        print( np.min(devy), end = ' ')
        print( np.max(devy), end = ' ')
        print( ','.join(list(set(devt))))

        print( len(testseqs), end=' ' )
        print( len(set(testtfs)), end=' ' )
        print( len(set(traintfs).intersection(set(testtfs))), end=' ')
        print( np.mean(list(c_test.values())), end=' ')
        print( np.std(list(c_test.values())), end=' ')
        print( np.median(list(c_test.values())), end=' ')
        print( np.min(list(c_test.values())), end = ' ')
        print( np.max(list(c_test.values())), end = ' ')
        print( np.min(testy), end = ' ')
        print( np.max(testy), end = ' ')
        print( ','.join(list(set(testt))))

        print()
