import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    h = query.size(1)
    d_k = query.size(-1)
    seq_len = query.size(-2)
    mask = mask.expand(-1, -1, seq_len, -1).expand(-1, h, -1, -1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -np.inf)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            # mask size: batch_size x 1 x 1 x seq_len
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # dim of q,k,v: batch_size x h x seq_len x d_k

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # dim of x: batch_size x h x seq_len x d_k
        # dim of self.attn: batch_size x h x seq_len x seq_len

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Embeddings(nn.Module):
    def __init__(self, we, vocab, embed_dim, freeze):
        super(Embeddings, self).__init__()
        word_embedding = torch.from_numpy(we).float()
        self.word_embeddings = nn.Embedding(vocab, embed_dim, _weight=word_embedding)
        self.position_embeddings = nn.Embedding(5000, embed_dim)
        if freeze:
            self.word_embeddings.weight.requires_grad = False
        # self.LayerNorm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)

        word_embeddings = self.word_embeddings(x)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Transformer(nn.Module):
    """
    A transformer encoder.
    """

    def __init__(self, args, config, word_embedding):
        super(Transformer, self).__init__()

        self.n_class = args.n_class
        embed_dim = 300
        d_model = int(config['d_model'])
        h = int(config['n_head'])
        d_ff = int(config['d_ff'])
        N = int(config['n_layer'])
        vocab_size = args.vocab_size

        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout=0.1)
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.1)
        layer = EncoderLayer(d_model, c(attn), c(ffn), 0.1)

        self.embed = Embeddings(word_embedding, vocab_size, embed_dim, args.freeze)
        self.linear = nn.Linear(embed_dim, d_model)
        self.encoder = Encoder(c(layer), N)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(d_model, d_ff)
        self.out = nn.Linear(d_ff, self.n_class)

        for name, p in self.named_parameters():
            if p.dim() > 1 and name != 'embed.word_embeddings.weight':
                # nn.init.xavier_uniform_(p)
                nn.init.normal_(p, 0, 0.05)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, mask):
        '''
        :param x: batch_size x seq_len
        :param y: batch_size
        :param mask: batch_size x seq_len
        :return:
            loss: scale
            pred: batch_size
        '''
        x = self.embed(x)
        x = self.linear(x)
        x = self.encoder(x, mask.unsqueeze(-2))
        # x: batch_size x seq_len x d_model  ->  batch_size x d_model
        x = x[:, 0, :]
        logit = self.out(self.dropout(F.relu(self.fc(x))))

        loss = self.criterion(logit, y)
        pred = torch.argmax(logit, dim=1)

        return loss, pred
