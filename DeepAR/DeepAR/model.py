#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020-07-10 23:15
# @Author : YuHui Li(MerylLynch)
# @File : model.py.py
# @Comment : Created By Liyuhui,23:15
# @Completed : No
# @Tested : No

import numpy as np
import torch as t


class Distribution(t.nn.Module):

    def __init__(self, input_size):
        super(Distribution, self).__init__()
        self.mu_layer = t.nn.Linear(input_size, 1)
        self.sigma_layer = t.nn.Linear(input_size, 1)

    def forward(self, rnn_hidden):
        mu = self.mu_layer(rnn_hidden)
        sigma = self.sigma_layer(rnn_hidden)
        return mu, sigma


class Attention(t.nn.Module):

    def __init__(self, Qdim, Kdim, Vdim, dim):
        super(Attention, self).__init__()
        self.linearQ = t.nn.Linear(Qdim, dim)
        self.linearK = t.nn.Linear(Kdim, dim)

    def forward(self, Q, K, V):
        Q = self.linearQ(Q.squeeze())
        K = self.linearK(K.permute(1, 0, 2))
        attn = t.bmm(K, Q.unsqueeze(-1))
        attn = t.softmax(attn, dim=1)
        output = t.bmm(attn.permute(0, 2, 1), V.permute(1, 0, 2)).squeeze()
        return output


class TPAMech(t.nn.Module):
    def __init__(self, Qdim, Kdim, Vdim, dim):
        super(TPAMech, self).__init__()
        self.linearQ = t.nn.Linear(Qdim, dim)
        self.linearK = t.nn.Linear(Kdim, dim)

    def forward(self, Q, K, V):
        # [bs,hidden] -> [bs, dim]
        Q = self.linearQ(Q.squeeze())
        # [bs,seq,feature] -> [bs,feature,seq]
        # [bs,feature,seq] -> [bs,feature,dim]
        K = self.linearK(K.permute(0, 2, 1))
        # K=[bs,feature,dim] Q=[bs,dim,1]
        attn = t.bmm(K, Q.unsqueeze(-1))
        attn = t.softmax(attn, dim=1)
        # TPA=[bs,feature,1] * V [bs,seqlen,feature]
        output = t.bmm(V, attn).squeeze()
        return output


class DeepAR(t.nn.Module):

    def __init__(self, z_size, feat_size, hidden_size,
                 num_layers, dropout=0.5,
                 forcast_step=12, encode_step=24, teacher_prob=0.5):
        super(DeepAR, self).__init__()
        self.forcast_step = forcast_step
        self.encode_step = encode_step
        self.teacher_prob = teacher_prob

        self.input_embed = t.nn.Linear(z_size, hidden_size)
        self.feat_embed = t.nn.Linear(feat_size, hidden_size)

        self.rnn_encoder = t.nn.GRU(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            dropout=dropout)
        self.encoder_norm = t.nn.LayerNorm(hidden_size)

        self.rnn_decoder = t.nn.GRU(
            input_size=2 * hidden_size + forcast_step,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            dropout=dropout
        )
        self.decoder_norm = t.nn.LayerNorm(hidden_size)
        self.attention = Attention(hidden_size, hidden_size,
                                   hidden_size, hidden_size)
        self.TPAMech = TPAMech(hidden_size, forcast_step, forcast_step, hidden_size)
        self.distribution = Distribution(hidden_size)
        self.output = t.nn.Linear(hidden_size, 1)

    def forward(self, histx, histz, futx, z):
        # Step 0 : Prepare Data
        futureZ = z[:, -self.forcast_step - 1:-1, ]

        # Step 1 : Encoder
        hidden = None
        enc_states = []
        for i in range(self.encode_step):
            zinput = self.input_embed(histz[:, i, :])
            xinput = self.feat_embed(histx[:, i, :])
            cell_input = t.cat((zinput, xinput), dim=-1).unsqueeze(0)
            output, hidden = self.rnn_encoder(cell_input, hidden)
            enc_states.append(output)

        enc_states = t.cat(enc_states, dim=0)

        # Step 2 : Decoder
        dec_states = []
        state = enc_states[-1, :, :, ].unsqueeze(0)
        for i in range(self.forcast_step):
            # static feature input
            featinput = self.feat_embed(futx[:, i, :]).unsqueeze(0)

            # last value input
            zinput = state
            if np.random.rand() < self.teacher_prob and self.training:
                zinput = self.input_embed(futureZ[:, i, :]).unsqueeze(0)

            # attention input
            attninp = self.attention(state, enc_states, enc_states).unsqueeze(0)
            # TPA Mech
            tpainput = self.TPAMech(state, futx, futx).unsqueeze(0)
            # Decoder Input
            dinput = t.cat((featinput, attninp, tpainput), dim=-1)
            state, hidden = self.rnn_decoder(dinput, hidden)
            dec_states.append(state)

        dec_states = t.cat(dec_states, dim=0)

        all_states = t.cat((enc_states, dec_states), dim=0)

        zhat = self.output(all_states.permute(1, 0, 2).relu())

        return zhat, zhat[:, -self.forcast_step:, :]
