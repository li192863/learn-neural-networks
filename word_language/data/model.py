"""
Transformer的简易实现原理
"""
import numpy as np
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    计算注意力权重，q、k、v必须具有匹配的前置维度且dq=dk，k、v必须有匹配的倒数第二个维度（seq_len_k = seq_len_v）
    :param q: 请求 (..., seq_len_q, depth)
    :param k: 主键 (..., seq_len_k, depth)
    :param v: 数值 (..., seq_len_v, depth_v)  seq_len_k = seq_len_v
    :param mask: 形状能转换成(..., seq_len_q, seq_len_k)，默认为None
    :return: 输出，注意力权重, (..., seq_len_q, depth_v), (..., seq_len_q, seq_len_k)
    """
    # matmul(a,b)矩阵乘（a b的最后2个维度要能做乘法）
    matmul_qk = torch.matmul(q, k.transpose(-1, -2))  # (..., seq_len_q, seq_len_k)
    # 缩放matmul_qk
    dk = torch.tensor(k.shape[-1], dtype=torch.float32)  # k的深度dk，或叫做depth_k
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)  # (..., seq_len_q, seq_len_k)
    # 将mask加入到缩放的张量上
    if mask is not None:  # (b, 1, 1, seq_len_k)
        scaled_attention_logits += (mask * -1e9)  # mask=1的位置是pad，-1e9相当于-∞，经过softmax后趋于0
    # 在最后一个轴seq_len_k上归一化
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights  # (..., seq_len_q, depth_v), (..., seq_len_q, seq_len_k)


def point_wise_feed_forward_network(d_model, dff):
    """
    点式前馈网络
    @param d_model: 输出最后维度
    @param dff: 隐藏层维度
    @return:
    """
     # x = (b, seq_len, d_model)
    feed_forward_net = torch.nn.Sequential(
        nn.Linear(d_model, dff),  # (b, seq_len, dff=2048)
        nn.ReLU(),
        nn.Linear(dff, d_model),  # (b, seq_len, d_model=512)
    )
    return feed_forward_net


def get_angles(pos, i, d_model):
    """
    计算角度 pos * 1/(10000^(2i/d))
    @param pos: position，(50, 1)
    @param i: d_model中第i个维度，(1, d_model)
    @param d_model: d_model
    @return: (50, d_model)
    """
    # 2 * (i//2)保证了2i，这部分计算的是1 / 10000^(2i / d)
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))  # (1, d_model=512)
    return pos * angle_rates  # (50, d_model=512)


def positional_encoding(position, d_model):
    """
    位置编码
    @param position: 位置
    @param d_model: 位置编码的长度，相当于position encoding的embedding_dim
    @return:
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],  #  (50, 1)
                            np.arange(d_model)[np.newaxis, :],  # (1, d_model=512)
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 2i: 0, 2, 4, 6, ...
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 2i+2: 1, 3, 5, 7, ...

    pos_encoding = angle_rads[np.newaxis, ...]  # (1, 50, 512)
    return torch.tensor(pos_encoding, dtype=torch.float32)


def create_padding_mask(seq, pad=1):  # (b, seq_len)
    """
    mask pad，即句子中为pad的位置处其mask值为1
    @param seq:
    @param pad:
    @return:
    """
    seq = torch.eq(seq, torch.tensor(pad)).float()  # pad!=0
    return seq[:, np.newaxis, np.newaxis, :]  # (b, 1, 1, seq_len)


def create_look_ahead_mask(size):  # seq_len
    """
    mask future token，将当前token后面的词mask掉，只让看到前面的词，即future token位置的mask值为1
    @param size:
    @return:
    """
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask  # (seq_len, seq_len)



class MultiHeadAttention(torch.nn.Module):
    """ 多头注意力 """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # 8
        self.d_model = d_model  # 512

        assert d_model % self.num_heads == 0  # 因为输入要被平均split到不同的head

        self.depth = d_model // self.num_heads  # 512/8=64，所以在scaled dot-product atten中dq=dk=64,dv也是64

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):  # (b, seq_len, d_model)
        x = x.view(batch_size, -1, self.num_heads, self.depth)  # (b, seq_len, num_head=8, depth=64)
        return x.transpose(1, 2)  # (b, num_head=8, seq_len, depth=64)

    def forward(self, q, k, v, mask):  # q=k=v=x, (b, seq_len, embedding_dim), embedding_dim=d_model
        batch_size = q.shape[0]

        q = self.wq(q)  # (b, seq_len, d_model)
        k = self.wk(k)  # (b, seq_len, d_model)
        v = self.wv(v)  # (b, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (b, num_head=8, seq_len, depth=64)
        k = self.split_heads(k, batch_size)  # (b, num_head=8, seq_len, depth=64)
        v = self.split_heads(v, batch_size)  # (b, num_head=8, seq_len, depth=64)

        # (b, num_head=8, seq_len_q, depth=64), (b, num_head=8, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.transpose(1, 2)  # (b, seq_len_q, num_head=8, depth=64)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)  # (b, seq_len_q, d_model=512)

        output = self.final_linear(concat_attention)  # (b, seq_len_q, d_model=512)
        return output, attention_weights  # (b, seq_len_q, d_model=512), (b, num_head=8, seq_len_q, seq_len_k)


class TransformerEncoderLayer(torch.nn.Module):
    """ 编码器层 """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)  # 多头注意力（padding mask）(self-attention)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)


    def forward(self, x, mask):  # (b, inp_seq_len, embedding_dim)(embedding_dim=d_model), (b, 1, 1, inp_seq_len)
        attn_output, _ = self.mha(x, x, x, mask)  # (b, inp_seq_len, d_model), (b, num_heads, inp_sql_len, inp_sql_len)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # 残差&层归一化 (b, inp_seq_len, d_model)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # 残差&层归一化 (b, inp_seq_len, d_model)
        return out2  # (b, inp_seq_len, d_model)


class TransformerDecoderLayer(torch.nn.Module):
    """ 解码器层 """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)  # masked的多头注意力（look ahead mask 和 padding mask）(self-attention)
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # 多头注意力（padding mask）(encoder-decoder attention)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)


    def forward(self, x, enc_output, look_ahead_mask, padding_mask):  #  (b, targ_seq_len, embedding_dim)(embedding_dim=d_model)
        # enc_output (b, inp_seq_len, d_model)
        # look_ahead_mask (b, 1, targ_seq_len, targ_seq_len) 这里传入的look_ahead_mask应该是已经结合了look_ahead_mask和padding mask的mask
        # padding_mask (b, 1, 1, inp_seq_len)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (b, targ_seq_len, d_model), (b, num_heads, targ_seq_len, targ_seq_len)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)  # 残差&层归一化 (b, targ_seq_len, d_model)

        # Q: receives the output from decoder's first attention block, which is masked multi-head attention sublayer
        # K, V: V (value) and K (key) receive the encoder output as inputs
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)  # (b, targ_seq_len, d_model), (b, num_heads, targ_seq_len, inp_seq_len)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)  # 残差&层归一化 (b, targ_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (b, targ_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)  # 残差&层归一化 (b, targ_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2  # (b, targ_seq_len, d_model), (b, num_heads, targ_seq_len, targ_seq_len), (b, num_heads, targ_seq_len, inp_seq_len)


class Encoder(torch.nn.Module):
    """ 编码器 """
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 input_vocab_size,  # 输入词表大小（源语言（法语））
                 maximum_position_encoding,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)  # (1, max_pos_encoding, d_model=512]

        self.embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)


    def forward(self, x, mask):  # (b, inp_seq_len), (b, 1, 1, inp_sel_len)
        inp_seq_len = x.shape[-1]

        # adding embedding and position encoding
        x = self.embedding(x)  # (b, inp_seq_len, d_model)
        # 缩放 embedding 原始论文的3.4节有提到： In the embedding layers, we multiply those weights by \sqrt{d_model}.
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_encoding = self.pos_encoding[:, :inp_seq_len, :]
        pos_encoding = pos_encoding.cuda()
        x += pos_encoding  # (b, inp_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)  # (b, inp_seq_len, d_model)
        return x  # (b, inp_seq_len, d_model)


class Decoder(torch.nn.Module):
    """ 解码器 """
    def __init__(self,
                 num_layers,  # N个decoder layer
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 target_vocab_size,  # target词表大小（目标语言（英语））
                 maximum_position_encoding,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)  # (1, max_pos_encoding, d_model=512)

        self.embedding = torch.nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=d_model)

        self.dec_layers = torch.nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

        self.dropout = torch.nn.Dropout(rate)


    def forward(self, x, enc_output, look_ahead_mask, padding_mask):  # (b, targ_seq_len)
        # enc_output (b, inp_seq_len, d_model)
        # look_ahead_mask (b, 1, targ_seq_len, targ_seq_len) 这里传入的look_ahead_mask应该是已经结合了look_ahead_mask和padding mask的mask
        # padding_mask (b, 1, 1, inp_seq_len)
        targ_seq_len = x.shape[-1]
        attention_weights = {}

        # adding embedding and position encoding
        x = self.embedding(x)  # (b, targ_seq_len, d_model)
        # 缩放 embedding 原始论文的3.4节有提到： In the embedding layers, we multiply those weights by \sqrt{d_model}.
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_encoding = self.pos_encoding[:, :targ_seq_len, :]  # (b, targ_seq_len, d_model)
        pos_encoding = pos_encoding.cuda()
        x += pos_encoding  # (b, inp_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attn_block1, attn_block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            # (b, targ_seq_len, d_model), (b, num_heads, targ_seq_len, targ_seq_len), (b, num_heads, targ_seq_len, inp_seq_len)
            attention_weights[f'decoder_layer{i + 1}_block1'] = attn_block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = attn_block2

        return x, attention_weights  # (b, targ_seq_len, d_model), {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],'..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}


class Transformer(torch.nn.Module):
    """ Transformer模型 """
    def __init__(self,
                 num_layers,  # N个encoder layer
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 input_vocab_size,  # input此表大小（源语言（法语））
                 target_vocab_size,  # target词表大小（目标语言（英语））
                 pe_input,  # input max_pos_encoding
                 pe_target,  # input max_pos_encoding
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               input_vocab_size,
                               pe_input,
                               rate)
        self.decoder = Decoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               target_vocab_size,
                               pe_target,
                               rate)
        self.final_layer = nn.Linear(d_model, target_vocab_size)


    def forward(self, inp, targ, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # inp (b, inp_seq_len)
        # targ (b, targ_seq_len)
        # enc_padding_mask (b, 1, 1, inp_seq_len)
        # look_ahead_mask (b, 1, targ_seq_len, targ_seq_len)
        # dec_padding_mask (b, 1, 1, inp_seq_len) # 注意这里的维度是inp_seq_len
        enc_output = self.encoder(inp, enc_padding_mask)  # (b, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder(targ, enc_output, look_ahead_mask, dec_padding_mask)  # (b, targ_seq_len, d_model), {'..block1': [b, num_heads, targ_seq_len, targ_seq_len], '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}
        final_output = self.final_layer(dec_output)  # (b, targ_seq_len, target_vocab_size)

        return final_output, attention_weights  # (b, targ_seq_len, target_vocab_size), {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}


if __name__ == '__main__':
    model = Transformer(6, 512, 8, 1024, 100, 500, 50, 50)
    print(model)
    padding_mask = create_padding_mask(torch.tensor([[2, 125, 44, 85, 231, 84, 130, 84, 1, 1, 1]]))
    print(padding_mask)
    print(padding_mask.shape)
    look_ahead_mask = create_look_ahead_mask(5)
    print(look_ahead_mask)
    print(look_ahead_mask.shape)
