import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
#from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.PeriodAttention import PeriodAttentionLayer, PeriodAttention
from layers.AutoSelectformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.slt_len = configs.slt_len
        self.slt_rate = configs.slt_rate
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


        # Eecoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    PeriodAttentionLayer(
                        PeriodAttention(configs.period, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    d_ff = configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers) ### 1
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.decoder = Encoder(
            [
                EncoderLayer(
                    PeriodAttentionLayer(
                        PeriodAttention(configs.period, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,

                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)  ### 1
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        scale = 4
        self.proj = nn.Sequential(
            nn.Linear(configs.enc_in, configs.enc_in * scale),
            nn.LayerNorm(configs.enc_in * scale),
            nn.ReLU(),
            nn.Linear(configs.enc_in * scale, 1)
        )

    def forward(self, x_en, x_mark_en, x_mark_de,
                en_self_mask=None, de_self_mask=None, de_enc_mask=None):
        # select
        en_select, en_mark = self._selecter(x_en,x_mark_en)

        # encoder
        en_trend = torch.zeros(en_select.shape).cuda()
        en_input = self.enc_embedding(en_select, en_mark)
        en_out, en_trend = self.encoder(en_input, mask=en_self_mask,trend = en_trend)
        # decoder init
        mean = torch.mean(en_trend, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        trend_init = torch.cat([en_trend,mean],dim=1)
        zeros = torch.zeros([en_select.shape[0], self.pred_len, en_select.shape[2]]).cuda()
        seasonal_init = torch.cat([en_select, zeros], dim=1)  ### [32, 144, 1]

        # decoder
        de_mark = torch.cat([en_mark,x_mark_de],1)
        de_input = self.dec_embedding(seasonal_init, de_mark)

        seasonal_part, trend_part = self.decoder(de_input, mask=de_self_mask, cross_mask=de_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    def _selecter(self, x, mark):
        return x[:,:self.slt_len,:], mark[:,:self.slt_len,:]

    def _selecter1(self, x, mark):
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