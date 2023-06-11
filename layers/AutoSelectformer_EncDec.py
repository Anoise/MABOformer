import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) ###复制一个像素
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    AutoSelectformer eecoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, d_model, c_out, cross_attention =None, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        if cross_attention is not None:
            self.cross_attention = cross_attention
            self.decomp2 = series_decomp(moving_avg)

        self.decomp1 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, mask=None, cross=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=mask
        )[0])
        x, trend = self.decomp1(x)
        if cross is not None and self.cross_attention is not None:
            x = x + self.dropout(self.cross_attention(
                x, cross, cross,
                attn_mask=cross_mask
            )[0])
            x, _trend = self.decomp2(x)
            trend += _trend
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, _trend = self.decomp3(x + y)
        #print(x.shape,y.shape,trend1.shape,trend2.shape,trend3.shape,' ttt ...')
        residual_trend = trend + _trend
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        #print(residual_trend.shape,'res shape')
        return x, residual_trend


class Encoder(nn.Module):
    """
    AutoSelectformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, mask=None, cross=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, mask= mask, cross=cross, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)
        #print(x.shape,'pre norm ...')
        if self.projection is not None:
            x = self.projection(x)
        #print(x.shape, 'pro norm ...')
        return x, trend
