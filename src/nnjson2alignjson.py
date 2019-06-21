import json
import os
import argparse

def readNNSeqs(filename):
    nnseqs = []
    with open(filename, 'r') as f:
        for line in f:
            nnseqs.append(json.loads(line))
    return nnseqs

def features2symbols(features):
    featnames = features.keys()
    symbols = []
    for i in range(len(features['beat'])):
        symbol = dict()
        for ft in featnames:
            symbol[ft] = features[ft][i]
        symbols.append(symbol)
    return symbols

def nnseq2alignseq(nnseq):
    alignseq = dict()
    alignseq[nnseq['id']] = dict()
    alignseq[nnseq['id']]['symbols'] = features2symbols(nnseq['features'])
    return alignseq

def nnseqs2alignseqs(nnseqs):
    alignseqs = []
    for nnseq in nnseqs:
        alignseqs.append(nnseq2alignseq(nnseq))
    return alignseqs

def writeSeqs(filepath, alignseqs):
    for alignseq in alignseqs:
        with open(f'{filepath}/{list(alignseq.keys())[0]}.json', 'w') as f:
            json.dump(alignseq, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert .jsonl to separate .json files for alignment')
    parser.add_argument('-inputfile', metavar='inputfile', type=str, help='.jsonl input file')
    parser.add_argument('-outputpath', metavar='outputpath', type=str, help='path to save output files', default='json/')
    args = parser.parse_args()

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    nnseqs = readNNSeqs(args.inputfile)
    alignseqs = nnseqs2alignseqs(nnseqs)
    writeSeqs(args.outputpath, alignseqs)
