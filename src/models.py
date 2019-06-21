import json
import os

import torch
import torch.nn.functional as F

import numpy as np


class CosinePairLoss(torch.nn.Module):
    def __init__(self, weight=0.25, margin=0.4, cutoff=False):
        super(CosinePairLoss, self).__init__()
        self.weight = weight
        self.margin = margin
        self.cutoff = cutoff

    def forward(self, x0, x1):
        return F.cosine_similarity(x0, x1)

    def loss(self, sims, y):
        pos = self.weight * (1 - sims) ** 2
        if self.cutoff:
            neg = (sims ** 2) * torch.gt(sims, self.margin).float()
        else:
            neg = torch.clamp(self.margin - (1 - sims), min=0) ** 2
        return torch.mean(y * pos + (1 - y) * neg)


class EuclideanPairLoss(torch.nn.Module):
    def __init__(self, weight=0.5, margin=1, cutoff=False):
        super(EuclideanPairLoss, self).__init__()
        self.margin = margin
        self.weight = weight
        self.cutoff = cutoff

    def forward(self, x0, x1):
        return F.pairwise_distance(x0, x1)

    def loss(self, dists, y):
        pos = self.weight * (dists ** 2)
        if self.cutoff:
            neg = (dists ** 2) * torch.gt(dists, self.margin).float()
        else:
            neg = torch.clamp(self.margin - dists, min=0) ** 2
        return torch.mean(y * pos + (1 - y) * neg)


class OnlineEuclideanPairLoss(torch.nn.Module):
    def __init__(self, pair_selector, weight=0.5, margin=1.0, cutoff=False):
        super(OnlineEuclideanPairLoss, self).__init__()
        self.margin = margin
        self.weight = weight
        self.pair_selector = pair_selector
        self.cutoff = cutoff

    def forward(self, embs, y):
        pos_pairs, neg_pairs = self.pair_selector.get_pairs(embs, y)
        if embs.is_cuda:
            pos_pairs = pos_pairs.cuda()
            neg_pairs = neg_pairs.cuda()

        # Compute loss for positives
        pos_dists = F.pairwise_distance(embs[pos_pairs[:, 0]], embs[pos_pairs[:, 1]])
        pos_loss = self.weight * (pos_dists ** 2)

        # Compute loss for negatives
        neg_dists = F.pairwise_distance(embs[neg_pairs[:, 0]], embs[neg_pairs[:, 1]])
        if self.cutoff:
            neg_loss = (neg_dists ** 2) * torch.gt(neg_dists, self.margin).float()
        else:
            neg_loss = torch.clamp(self.margin - neg_dists, min=0) ** 2

        loss = torch.cat([pos_loss, neg_loss], dim=0)
        return loss.mean()

    def loss(self, embs, y):
        return self(embs, y)


class OnlineCosinePairLoss(torch.nn.Module):
    def __init__(self, pair_selector, weight=0.5, margin=0.5, cutoff=False):
        super(OnlineCosinePairLoss, self).__init__()
        self.weight = weight
        self.margin = margin
        self.pair_selector = pair_selector
        self.cutoff = cutoff

    def forward(self, embs, y):
        pos_pairs, neg_pairs = self.pair_selector.get_pairs(embs, y)
        if embs.is_cuda:
            pos_pairs = pos_pairs.cuda()
            neg_pairs = neg_pairs.cuda()

        # Compute loss for positives
        pos_sims = F.cosine_similarity(embs[pos_pairs[:, 0]], embs[pos_pairs[:, 1]])
        # Compute loss for negatives
        neg_sims = F.cosine_similarity(embs[neg_pairs[:, 0]], embs[neg_pairs[:, 1]])
        return pos_sims, neg_sims

    def loss(self, embs, y):
        pos_sims, neg_sims = self(embs, y)
        pos_loss = self.weight * (1 - pos_sims) ** 2

        if self.cutoff:
            neg_loss = (neg_sims ** 2) * torch.gt(neg_sims, self.margin).float()
        else:
            neg_loss = torch.clamp(self.margin - (1 - neg_sims), min=0) ** 2

        loss = torch.cat([pos_loss, neg_loss], dim=0)
        return loss.mean()


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        raise NotImplementedError

    def loss(self, pos_dists, neg_dists):
        return torch.clamp(pos_dists - neg_dists + self.margin, min=0).mean()


class EuclidianTripletLoss(TripletLoss):

    def forward(self, anchor, positive, negative):
        pos_dists = F.pairwise_distance(anchor, positive, 2)
        neg_dists = F.pairwise_distance(anchor, negative, 2)
        return pos_dists, neg_dists


class CosineTripletLoss(TripletLoss):
    def __init__(self, margin=0.5):
        super(CosineTripletLoss, self).__init__(margin=margin)

    def forward(self, anchor, positive, negative):
        pos_dists = -F.cosine_similarity(anchor, positive)
        neg_dists = -F.cosine_similarity(anchor, negative)
        return pos_dists, neg_dists


class OnlineTripletLoss(torch.nn.Module):
    def __init__(self, triplet_selector, margin=1):
        super(OnlineTripletLoss, self).__init__()
        self.triplet_selector = triplet_selector
        self.margin = margin

    def forward(self, anchor, positive, negative):
        raise NotImplementedError

    def loss(self, embs, y):
        triplets = self.triplet_selector.get_triplets(embs, y)
        if embs.is_cuda:
            triplets = triplets.cuda()
        pos_dists, neg_dists = self(
            embs[triplets[:, 0]], embs[triplets[:, 1]], embs[triplets[:, 2]])
        return torch.clamp(pos_dists - neg_dists + self.margin, min=0).mean()


class OnlineEuclideanTripletLoss(OnlineTripletLoss):

    def forward(self, anchor, positive, negative):
        pos_dists = F.pairwise_distance(anchor, positive)
        neg_dists = F.pairwise_distance(anchor, negative)
        return pos_dists, neg_dists


class OnlineCosineTripletLoss(OnlineTripletLoss):
    def __init__(self, triplet_selector, margin=0.5):
        super(OnlineCosineTripletLoss, self).__init__(triplet_selector, margin=margin)

    def forward(self, anchor, positive, negative):
        pos_dists = -F.cosine_similarity(anchor, positive)
        neg_dists = -F.cosine_similarity(anchor, negative)
        return pos_dists, neg_dists


def init_conv(conv):
    conv.reset_parameters()
    torch.nn.init.xavier_uniform_(conv.weight)
    torch.nn.init.constant_(conv.bias, 0.)


def init_embeddings(embeddings):
    embeddings.reset_parameters()
    torch.nn.init.constant_(embeddings.weight, 0.01)


class Highway(torch.nn.Module):
    def __init__(self, input_dim, num_layers=1, activation=torch.nn.functional.relu):
        super(Highway, self).__init__()
        self.__input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class CNN(torch.nn.Module):
    def __init__(self, emb_dims, cont_features=0, dropout=0.5,
                 kernel_sizes=(5, 4, 3), out_channels=32,
                 highway_layers=2, default_init=True):
        self.dropout = dropout
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        super(CNN, self).__init__()

        self.embs = torch.nn.ModuleList([
            torch.nn.Embedding(emb_num, emb_dim, padding_idx=0)
            for emb_num, emb_dim in emb_dims
        ])
        
        convs = []
        embedding_dim = sum(emb_dim for _, emb_dim in emb_dims) + cont_features
        for W in kernel_sizes:
            padding = ((W//2) - 1, W - (W//2), 0, 0)
            conv = torch.nn.Sequential(
                torch.nn.ZeroPad2d(padding),
                torch.nn.Conv2d(1, out_channels, (embedding_dim, W)))
            convs.append(conv)
        self.convs = torch.nn.ModuleList(convs)

        # highway layers
        self.highway = None
        if highway_layers > 0:
            self.highway = Highway(len(kernel_sizes) * out_channels, highway_layers)

        if not default_init:
            for embedding in self.embs:
                init_embeddings(embedding)

            for conv in self.convs:
                init_conv(conv)

    def device(self):
        return next(self.parameters()).device

    def forward(self, X, lengths):
        _, unsort = lengths
        cat_X, cont_X = X
        if cat_X is not None:
            emb = [emb(cat_X[:, i]) for i, emb in enumerate(self.embs)]
            emb = torch.cat(emb, 2)
        if cont_X is not None:
            cont_X = cont_X.permute(0, 2, 1)
            if cat_X is not None:
                emb = torch.cat([emb, cont_X], 2)
            else:
                emb = cont_X
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        emb = emb.transpose(1, 2)
        emb = emb.unsqueeze(1)

        conv_outs, maxlen = [], 0
        for conv in self.convs:
            # (batch x C_o x seq_len)
            conv_outs.append(F.relu(conv(emb).squeeze(2)))

        conv_outs = torch.cat(conv_outs, dim=1)
        conv_outs = F.max_pool1d(conv_outs, conv_outs.size(2)).squeeze(2)
        if self.highway is not None:
            conv_outs = self.highway(conv_outs)

        return conv_outs[unsort]

    def get_embedding(*features):
        return self(*features)


class CNNRNN(torch.nn.Module):
    def __init__(self, emb_dims, cont_features=0, dropout=0.5, default_init=True,
                 # CNN
                 kernel_sizes=(5, 4, 3), out_channels=32, highway_layers=2,
                 # RNN
                 cell='GRU', n_layers=1, bidirectional=True, forget_bias=False,
                 rnn_dropout=0.0):
        super(CNNRNN, self).__init__()

        self.embs = torch.nn.ModuleList([
            torch.nn.Embedding(emb_num, emb_dim, padding_idx=0)
            for emb_num, emb_dim in emb_dims
        ])

        convs = []
        embedding_dim = sum(emb_dim for _, emb_dim in emb_dims) + cont_features
        for W in kernel_sizes:
            padding = ((W//2) - 1, W - (W//2), 0, 0)
            conv = torch.nn.Sequential(
                torch.nn.ZeroPad2d(padding),
                torch.nn.Conv2d(1, out_channels, (embedding_dim, W)))
            convs.append(conv)
        self.convs = torch.nn.ModuleList(convs)

        output_dim = len(kernel_sizes) * out_channels

        # highway layers
        self.highway = None
        if highway_layers > 0:
            self.highway = Highway(output_dim, highway_layers)

        self.rnn = getattr(torch.nn, cell)
        self.rnn = torch.nn.LSTM(
            output_dim, output_dim // (1 + bidirectional), num_layers=n_layers,
            bidirectional=bidirectional, dropout=(n_layers > 1) * rnn_dropout,
            batch_first=True)

        if not default_init:
            for embedding in self.embs:
                init_embeddings(embedding)

            for conv in self.convs:
                init_conv(conv)

        if forget_bias and cell == 'LSTM':
            for pname, p in self.rnn.named_parameters():
                if 'bias' in pname:
                    torch.nn.init.constant_(p, 0.0)
                    n = p.size(0)
                    torch.nn.init.constant_(p[n//4:n//2], 1.0)

        self.dropout = dropout
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.bidirectional = bidirectional
        self.n_layers = n_layers

    def device(self):
        return next(self.parameters()).device

    def forward(self, X, lengths):
        lengths, unsort = lengths
        cat_X, cont_X = X
        if cat_X is not None:
            emb = [emb(cat_X[:, i]) for i, emb in enumerate(self.embs)]
            emb = torch.cat(emb, 2)
        if cont_X is not None:
            cont_X = cont_X.permute(0, 2, 1)
            if cat_X is not None:
                emb = torch.cat([emb, cont_X], 2)
            else:
                emb = cont_X
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        emb = emb.transpose(1, 2)
        emb = emb.unsqueeze(1)

        conv_outs, maxlen = [], 0
        for conv in self.convs:
            # (batch x C_o x seq_len)
            conv_outs.append(F.relu(conv(emb).squeeze(2)))

        conv_outs = torch.cat(conv_outs, dim=1)
        # (batch x seq_len x C_o)
        conv_outs = conv_outs.permute(0, 2, 1).contiguous()

        if self.highway is not None:
            batch, seq_len, C_o = conv_outs.size()
            conv_outs = self.highway(conv_outs.view(-1, C_o))
            conv_outs = conv_outs.view(batch, seq_len, C_o)

        conv_outs = torch.nn.functional.dropout(
            conv_outs, p=self.dropout, training=self.training)

        conv_outs = torch.nn.utils.rnn.pack_padded_sequence(
            conv_outs, lengths, batch_first=True)

        out, emb = self.rnn(conv_outs)
        if isinstance(emb, tuple):
            emb, _ = emb
        if self.bidirectional:
            # (num_layers * num_directions x batch x hidden_size)
            emb = emb.view(self.n_layers, self.bidirectional + 1, len(lengths), -1)
            emb = emb[-1]  # take last layer
            emb = torch.cat([emb[-2], emb[-1]], 1)
            return emb[unsort]
        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # (batch x seqlen x hid_dim) => (batch x hid_dim)
            out = out.sum(1) / lengths.float()[:, None].to(self.device())
            return out[unsort]


    def get_embedding(*features):
        return self(*features)


class RNN(torch.nn.Module):
    def __init__(self, emb_dims, hid_dim, cell='LSTM', cont_features=0,
                 forget_bias=False, n_layers=2, dropout=0.5, rnn_dropout=0,
                 bidirectional=False):
        super(RNN, self).__init__()

        self.embs = torch.nn.ModuleList([
            torch.nn.Embedding(emb_num, emb_dim, padding_idx=0)
            for emb_num, emb_dim in emb_dims
        ])
        
        self.rnn = getattr(torch.nn, cell)(
            sum(emb_dim for _, emb_dim in emb_dims) + cont_features,
            hid_dim // (1 + bidirectional), n_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=(n_layers > 1) * rnn_dropout)

        if forget_bias and cell == 'LSTM':
            for pname, p in self.rnn.named_parameters():
                if 'bias' in pname:
                    torch.nn.init.constant_(p, 0.0)
                    n = p.size(0)
                    torch.nn.init.constant_(p[n//4:n//2], 1.0)

        self.cell = cell
        self.emb_dims = emb_dims
        self.hid_dim = hid_dim
        self.cont_features = cont_features
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

    def get_arguments(self):
        return (self.emb_dims, self.hid_dim), {'cont_features': self.cont_features,
                                               'cell': self.cell,
                                               'n_layers': self.n_layers,
                                               'bidirectional': self.bidirectional}

    def save(self, filename):
        prefix, ext = os.path.splitext(filename)
        with open(prefix + '.json', 'w') as f:
            args, kwargs = self.get_arguments()
            f.write(json.dumps({'args': args, 'kwargs': kwargs}))
        with open(prefix + '.pt', 'wb') as f:
            torch.save(self.state_dict(), f)

    @classmethod
    def load(cls, filename, **kwargs):
        """
        - kwargs: arguments to override from serialized object (e.g. dropout)
        """
        prefix, ext = os.path.splitext(filename)
        with open(prefix + '.json') as f:
            arguments = json.loads(f.read())
        inst = cls(*arguments['args'], **dict(**arguments['kwargs'], **kwargs))
        inst.load_state_dict(torch.load(prefix + '.pt', map_location=inst.device()))
        return inst

    def device(self):
        return next(self.parameters()).device

    def forward(self, X, lengths):
        lengths, unsort = lengths
        cat_X, cont_X = X
        if cat_X is not None:
            emb = [emb(cat_X[:, i]) for i, emb in enumerate(self.embs)]
            emb = torch.cat(emb, 2)
        if cont_X is not None:
            cont_X = cont_X.permute(0, 2, 1)
            if cat_X is not None:
                emb = torch.cat([emb, cont_X], 2)
            else:
                emb = cont_X
        emb = torch.nn.functional.dropout(emb, p=self.dropout, training=self.training)
        emb = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
        out, emb = self.rnn(emb)
        if isinstance(emb, tuple):
            emb, _ = emb
        if self.bidirectional:
            # (num_layers * num_directions x batch x hidden_size)
            emb = emb.view(self.n_layers, self.bidirectional + 1, len(lengths), -1)
            emb = emb[-1]  # take last layer
            emb = torch.cat([emb[-2], emb[-1]], 1)
            return emb[unsort]

        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # (batch x seqlen x hid_dim) => (batch x hid_dim)
            out = out.sum(1) / lengths.float()[:, None].to(self.device())
            return out[unsort]

    def get_embedding(*features):
        return self(*features)


class Network(torch.nn.Module):
    def __init__(self, network, objective):
        super(Network, self).__init__()
        self.network = network
        self.objective = objective

    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.network(x[0], x[1])

    def get_embedding(self, *x):
        return self.network(*x)


class TwinNetwork(Network):
    def __init__(self, network, objective):
        super(TwinNetwork, self).__init__(network, objective)

    def forward(self, x0, x1):
        output0 = self.network(*x0)
        output1 = self.network(*x1)
        return self.objective(output0, output1)


class TripletNetwork(Network):
    def __init__(self, network, objective):
        super(TripletNetwork, self).__init__(network, objective)

    def forward(self, anchor, positive, negative):
        anchor_out = self.network(*anchor)
        positive_out = self.network(*positive)
        negative_out = self.network(*negative)
        return self.objective(anchor_out, positive_out, negative_out)
