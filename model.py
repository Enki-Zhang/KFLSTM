import math

import numpy as np
import torch as t


class Distribution(t.nn.Module):

    def __init__(self, input_size):
        super(Distribution, self).__init__()
        self.mu_layer = t.nn.Linear(input_size, 1)  # 设置全连接层 参数(输入神经元个数，输出神经元个数)
        self.sigma_layer = t.nn.Linear(input_size, 1)

    def forward(self, rnn_hidden):
        mu = self.mu_layer(rnn_hidden)
        sigma = self.sigma_layer(rnn_hidden)
        return mu, sigma


# class Attention(t.nn.Module):
#
#     def __init__(self, Qdim, Kdim, Vdim, dim):
#         super(Attention, self).__init__()
#         self.linearQ = t.nn.Linear(Qdim, dim)
#         self.linearK = t.nn.Linear(Kdim, dim)
#
#     def forward(self, Q, K, V):
#         Q = self.linearQ(Q.squeeze())
#         K = self.linearK(K.permute(1, 0, 2))
#         attn = t.bmm(K, Q.unsqueeze(-1))  # Q和K的转置相乘
#         attn = t.softmax(attn, dim=1)  # 归一化
#         output = t.bmm(attn.permute(0, 2, 1), V.permute(1, 0, 2)).squeeze()  # 和V相乘
#         return output


# 自注意力机制
class Attention(t.nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, Qdim, Kdim, Vdim, dim):
        super(Attention, self).__init__()
        self.dim_q = Kdim  # 一般默认 Q=K
        self.dim_k = Kdim
        self.dim_v = Vdim

        # 定义线性变换函数
        self.linear_q = t.nn.Linear(Qdim, dim, bias=False)
        self.linear_k = t.nn.Linear(Qdim, dim, bias=False)
        self.linear_v = t.nn.Linear(Qdim, dim, bias=False)
        self._norm_fact = 1 / math.sqrt(Kdim)

    def forward(self, Q, K, V):
        # x: batch_size, seq_len, input_dim

        q = self.linear_q(Q.squeeze())  # batch_size, seq_len, dim_k
        k = self.linear_k(K.permute(1, 0, 2))  # batch_size, seq_len, dim_k
        # v = self.linear_v(x)  # batch_size, seq_len, dim_v
        # q*k的转置 并*开根号后的dk
        dist = t.bmm(k, q.unsqueeze(-1))  # batch_size, seq_len, seq_len
        # 归一化获得attention的相关系数  对每个字求sofmax 也就是每一行
        dist = t.softmax(dist, dim=1)  # batch_size, seq_len, seq_len
        # attention系数和v相乘，获得最终的得分
        att = t.bmm(dist.permute(0, 2, 1), V.permute(1, 0, 2)).squeeze()
        return att


# class LayerNorm(t.nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         """Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(LayerNorm, self).__init__()
#         self.weight = t.nn.Parameter(t.ones(hidden_size))
#         self.bias = t.nn.Parameter(t.zeros(hidden_size))
#         self.variance_epsilon = eps
#
#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / t.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias
#
#
# class SelfAttention(t.nn.Module):
#     def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
#         super(SelfAttention, self).__init__()
#         if hidden_size % num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (hidden_size, num_attention_heads))
#         # 设超参数num_attention_heads为自注意力机制的头数，如此，计算出每个头的维度attention_head_size。
#         self.num_attention_heads = num_attention_heads
#         self.attention_head_size = int(hidden_size / num_attention_heads)
#         self.all_head_size = hidden_size
#         #  QKV
#         self.query = t.nn.Linear(input_size, self.all_head_size)
#         self.key = t.nn.Linear(input_size, self.all_head_size)
#         self.value = t.nn.Linear(input_size, self.all_head_size)
#
#         self.attn_dropout = t.nn.Dropout(0.5)  # attention_probs_dropout_prob == 0.5
#
#         # 做完self-attention 做一个前馈全连接 LayerNorm 输出
#         self.dense = t.nn.Linear(hidden_size, hidden_size)
#         self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
#         self.out_dropout = t.nn.Dropout(hidden_dropout_prob)
#
#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)
#
#     def forward(self, input_tensor):
#         mixed_query_layer = self.query(input_tensor)
#         mixed_key_layer = self.key(input_tensor)
#         mixed_value_layer = self.value(input_tensor)
#
#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)
#
#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = t.matmul(query_layer, key_layer.transpose(-1, -2))
#
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         # [batch_size heads seq_len seq_len] scores
#         # [batch_size 1 1 seq_len]
#
#         # attention_scores = attention_scores + attention_mask
#
#         # Normalize the attention scores to probabilities.
#         attention_probs = t.nn.Softmax(dim=-1)(attention_scores)
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         # Fixme
#         attention_probs = self.attn_dropout(attention_probs)
#         context_layer = t.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         hidden_states = self.dense(context_layer)
#         hidden_states = self.out_dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#
#         return hidden_states
#

# class MultiHeadedAttention(t.nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadedAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = t.clones(t.nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = t.nn.Dropout(p=dropout)
#
#     def forward(self, query, key, value, mask=None):
#         "Implements Figure 2"
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1)
#         nbatches = query.size(0)
#
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         # 先对输入值x进行reshape一下，然后交换在维度1,2进行交换
#         query, key, value = \
#             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip(self.linears, (query, key, value))]
#
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = attention(query, key, value, mask=mask,
#                                  dropout=self.dropout)
#
#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous() \
#             .view(nbatches, -1, self.h * self.d_k)
#         return self.linears[-1](x)/


class TPAMech(t.nn.Module):
    def __init__(self, Qdim, Kdim, Vdim, dim):
        super(TPAMech, self).__init__()
        self.linearQ = t.nn.Linear(Qdim, dim)
        self.linearK = t.nn.Linear(Kdim, dim)

    def forward(self, Q, K, V):
        # [bs,hidden] -> [bs, dim]
        # print(Q.shape,K.shape)
        Q = self.linearQ(Q.squeeze())
        # [bs,seq,feature] -> [bs,feature,seq]
        # [bs,feature,seq] -> [bs,feature,dim]
        K = self.linearK(K.permute(0, 2, 1))
        # print(Q.shape,K.shape)
        # K=[bs,feature,dim] Q=[bs,dim,1]
        attn = t.bmm(K, Q.unsqueeze(-1))
        # print(K.shape, Q.unsqueeze(-1).shape, attn.shape)
        attn = t.softmax(attn, dim=1)
        # TPA=[bs,feature,1] * V [bs,seqlen,feature]
        output = t.bmm(V, attn).squeeze()
        return output


class FM_layer(t.nn.Module):
    def __init__(self, n, k):
        super(FM_layer, self).__init__()
        self.n = n
        self.k = k
        self.linear = t.nn.Linear(self.n, 1)  # 前两项线性层
        self.v = t.nn.Parameter(t.randn(self.n, self.k))  # 交互矩阵
        t.nn.init.uniform_(self.v, -0.1, 0.1)

    def fm_layer(self, x):
        # print('x.shape = ', x.shape)
        linear_part = self.linear(x)
        # print('linear_part.shape:',linear_part.shape)
        interaction_part_1 = t.mm(x, self.v)
        # print('interaction_part_1',interaction_part_1.shape)
        interaction_part_1 = t.pow(interaction_part_1, 2)
        # print(self.v,self.v.shape)
        interaction_part_2 = t.mm(t.pow(x, 2), t.pow(self.v, 2))
        # print('interaction_part_2',interaction_part_2.shape)
        output = linear_part + 0.5 * t.sum(interaction_part_2 - interaction_part_1, 1, keepdim=False).unsqueeze(1)
        # print('output.shape:',output.shape)
        return output

    def forward(self, x):
        return self.fm_layer(x)


class DeepAR(t.nn.Module):

    def __init__(self, z_size, feat_size, hidden_size,
                 num_layers, dropout=0.5,
                 forcast_step=12, encode_step=24, teacher_prob=0.5):  # fm_k=72
        super(DeepAR, self).__init__()

        # self.fm = FM_layer(4,3)
        # print(fm_k)
        # fm
        # self.fm = FM_layer(hidden_size, fm_k)

        self.forcast_step = forcast_step
        self.encode_step = encode_step
        self.teacher_prob = teacher_prob

        self.input_embed = t.nn.Linear(z_size, hidden_size)
        self.feat_embed = t.nn.Linear(feat_size, hidden_size)  # 设置全连接层 [输入张量大小，输出张量大小]
        # self.rnn_encoder = t.nn.GRU(
        #     # input_size=257,  # 2 * hidden_size + 1
        #     input_size=2 * hidden_size ,  # 2 * hidden_size + 1
        #     # input_size=2 * hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     bias=True,
        #     dropout=dropout)

        self.rnn_encoder = t.nn.LSTM(
            # input_size=257,  # 2 * hidden_size + 1
            input_size=2 * hidden_size,  # 2 * hidden_size
            # input_size=2 * hidden_size,
            hidden_size=hidden_size,  # 128 默认值
            num_layers=num_layers,
            bias=True,
            # bidirectional=True,
            dropout=dropout, )
        self.encoder_norm = t.nn.LayerNorm(hidden_size)

        # self.rnn_decoder = t.nn.GRU(
        #     input_size=2 * hidden_size + forcast_step,  # 2 * hidden_size + forcast_step + 1,
        #     # input_size=269,  # 2 * hidden_size + forcast_step + 1,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     bias=True,
        #     dropout=dropout
        # )

        #
        self.rnn_decoder = t.nn.LSTM(
            input_size=2 * hidden_size + forcast_step,  # 2 * hidden_size + forcast_step + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            dropout=dropout
        )
        # self.decoder_norm = LayerNorm(hidden_size)
        self.decoder_norm = t.nn.LayerNorm(hidden_size)
        self.attention = Attention(hidden_size, hidden_size, hidden_size, hidden_size)
        # self.attention = SelfAttention(2, hidden_size, hidden_size, 0.5)
        self.TPAMech = TPAMech(hidden_size, forcast_step, forcast_step, hidden_size)
        self.distribution = Distribution(hidden_size)
        self.output = t.nn.Linear(hidden_size, 1)
        self.dist = Distribution(hidden_size)

    def forward(self, histx, histz, futx, z):
        # print('rowfea',rowfea.shape)
        # fm_res = self.fm(rowfea)
        # print(rowfea)
        # fm_res = self.fm(rowfea)
        # print('fm_res',fm_res.shape)
        # Step 0 : Prepare Data
        futureZ = z[:, -self.forcast_step - 1:-1, ]  # 取出预测的label长度

        # Step 1 : Encoder
        hidden = None
        enc_states = []
        for i in range(self.encode_step):
            zinput = self.input_embed(histz[:, i, :])  # zinput 取出历史label第二个维度的第i个
            xinput = self.feat_embed(histx[:, i, :])
            cell_input = t.cat((zinput, xinput), dim=1).unsqueeze(0)  # 拼接[1, 8, 256]
            # 输入数据处理完成
            output, hidden = self.rnn_encoder(cell_input, hidden)
            enc_states.append(output)

        enc_states = t.cat(enc_states, dim=0)

        # Step 2 : Decoder
        dec_states = []
        state = enc_states[-1, :, :, ].unsqueeze(0)  # this state is the h of the last time step      so called C
        for i in range(self.forcast_step):
            # static feature input
            featinput = self.feat_embed(futx[:, i, :]).unsqueeze(0)
            # fm
            # fm_input2 = self.fm(self.feat_embed(futx[:, i, :])).unsqueeze(0)

            # last value input
            zinput = state  # 模型在某些预测步上直接使用上一步的预测值作为输入 提高模型的稳定性
            if np.random.rand() < self.teacher_prob and self.training:
                zinput = self.input_embed(futureZ[:, i, :]).unsqueeze(0)
                # print(zinput.shape)

            # attention input
            attninp = self.attention(state, enc_states, enc_states).unsqueeze(0)
            # print('attn',attninp.shape)
            # TPA Mech
            tpainput = self.TPAMech(state, futx, futx).unsqueeze(0)
            # print('tpa', tpainput.shape)
            # Decoder Input
            # print(featinput.shape, attninp.shape, tpainput.shape, fm_input2.shape)
            # print(featinput.shape)
            # dinput = t.cat((featinput, attninp, tpainput, fm_input2), dim=-1)
            dinput = t.cat((featinput, attninp, tpainput), dim=-1)
            # print('featinput:',featinput.shape, 'attninp:', attninp.shape, 'tpainput:', tpainput.shape, 'fm_input2:', fm_input2.shape, 'dinput:', dinput.shape)
            # 将encoder的输出经过attention层 输入到decoder
            state, hidden = self.rnn_decoder(dinput, hidden)
            dec_states.append(state)

        dec_states = t.cat(dec_states, dim=0)

        all_states = t.cat((enc_states, dec_states), dim=0)

        zhat = self.output(all_states.permute(1, 0, 2).relu())

        # mu, sigma = self.dist(all_states.permute(1, 0, 2))
        # mu = t.sigmoid(mu)
        # sigma = t.sigmoid(sigma)

        return zhat, zhat[:, -self.forcast_step:]
