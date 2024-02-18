import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.PeriodAttentionV2 import PeriodAttentionLayer, PeriodAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import numpy as np

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2306.05035
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        # self.encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             PeriodAttentionLayer(
        #                 PeriodAttention(configs.period, attention_dropout=configs.dropout,
        #                                 output_attention=configs.output_attention),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             moving_avg=configs.moving_avg,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for _ in range(configs.e_layers) ### 2
        #     ],
        #     norm_layer=my_Layernorm(configs.d_model)
        # )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    PeriodAttentionLayer(
                        PeriodAttention(configs.period, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    PeriodAttentionLayer(
                        PeriodAttention(configs.period, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers) ### 1
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # scale = 4
        # self.proj = nn.Sequential(
        #     nn.Linear(configs.enc_in, configs.enc_in * scale),
        #     nn.LayerNorm(configs.enc_in * scale),
        #     nn.ReLU(),
        #     nn.Linear(configs.enc_in * scale, 1)
        # )

        # pooling_mode = 'max'
        # if pooling_mode == 'max':
        #     self.pooling_layer = nn.MaxPool1d(kernel_size=8,
        #                                       stride=8, ceil_mode=True)
        # elif pooling_mode == 'average':
        #     self.pooling_layer = nn.AvgPool1d(kernel_size=self.n_pool_kernel_size,
        #                                       stride=self.n_pool_kernel_size, ceil_mode=True)

    def forward(self, x_enc, x_mark_enc, x_mark_dec, test=False,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # select
        #x_enc, x_mark_enc = self._selecter(x_enc, x_mark_enc, test=test)
        #x_enc = self._select_channel(x_enc, test=test)
        # x_pool = self.pooling_layer(x_enc.transpose(1,2))
        # x_exp = F.interpolate(x_pool, size=self.pred_len, mode='linear', align_corners=True)
        # x_exp = x_exp.transpose(1,2)

        # decoder input
        trend_init = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]]).cuda()
        # enc
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, cross = None, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


    def _selecter(self, x, mark,test=False):
        a = self.slt_len
        return x[:,-a:,:], mark[:,-a:,:]

    def _select_channel(self, x,test=False):
        a = 6
        B,S,C = x.size()
        padding = torch.zeros(B,S,C-a).cuda()
        x = torch.cat([padding,x[:,:,-a:]],-1)
        return x

    def _selecter2(self, x, mark):
        B,S,C = x.size()
        score = self.proj(x)
        score = F.softmax(score.squeeze(-1), -1)
        _, x_idx = torch.topk(score, k=self.slt_len, dim = -1)
        series = torch.Tensor([[i*S] for i in range(B)]).cuda()
        x_idx = (series + x_idx).long()
        x = x.view(-1,C)[x_idx]
        x = x.view(B,-1,C)

        B, _S, C = mark.size()
        assert S == _S
        mark = mark.view(-1,C)[x_idx]
        mark = mark.view(B,-1,C)
        return x, mark
