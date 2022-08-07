import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


class quant_lookup(nn.Module):
    def __init__(self, size, granu, n_bits, is_act=True):
        super(quant_lookup, self).__init__()
        self.n_bits = n_bits
        self.is_act = is_act
        self.granu = granu
        self.size = size

        if is_act:
            self.scale = nn.Parameter(torch.tensor(0.0))
            self.range = 2 ** n_bits - 1
            T = torch.ones(1, 1, 2 * (granu * self.range) - 1)
            T[0, 0, (granu * self.range):] = 0
        else:
            self.scale = nn.Parameter(torch.tensor(0.0))
            if n_bits == 1:
                self.range = 1
                T = torch.ones(1, 1, 2 * (granu * self.range) - 1)
                T[0, 0, (granu * self.range):] = 0
            else:
                self.range = 2 ** (n_bits - 1) - 1
                T = torch.ones(1, 1, 2 * (granu * self.range) - 1)
                T[0, 0, (granu * self.range):] = 0

        self.table = nn.Parameter(torch.zeros(self.range, granu))
        self.register_buffer('T', T)

    def _update_tau(self, tau):
        self.tau = tau

    def _gen_table(self):
        if self.training:
            prob = (self.table / self.tau).softmax(1)
            self.loss_q = (1 - (prob ** 2).sum(1)).mean()
            prob = prob.view(1, 1, -1)
            table_q = F.conv1d(prob, self.T, padding=prob.size(-1) - 1).unsqueeze(-1)
            if self.is_act:
                table_q = F.pad(table_q, [0, 0, table_q.size(2) + 1, 0])
            else:
                if self.n_bits == 1:
                    table_q = torch.cat([-torch.ones(1,1,1,1).to(prob.device), table_q * 2 - 1], 2)
                else:
                    table_q = torch.cat([-table_q.flip(2), F.pad(table_q, [0, 0, 1, 0])], 2)

            return table_q / self.range

        else:
            index = self.table.max(1, keepdim=True)[1]
            prob = torch.zeros_like(self.table).scatter_(1, index, 1.0)
            prob = prob.view(1, 1, -1)
            table_q = F.conv1d(prob, self.T, padding=prob.size(-1) - 1).unsqueeze(-1)
            if self.is_act:
                table_q = F.pad(table_q, [0, 0, table_q.size(2) + 1, 0])
            else:
                if self.n_bits == 1:
                    table_q = torch.cat([-torch.ones(1, 1, 1, 1).to(prob.device), table_q * 2 - 1], 2)
                else:
                    table_q = torch.cat([-table_q.flip(2), F.pad(table_q, [0, 0, 1, 0])], 2)

            return table_q / self.range

    def _lookup(self, x, table_q, scale):
        if self.training:
            grid = (x / scale).clamp(-1, 1)
            if self.is_act:
                wgt = torch.histc(grid.data, table_q.numel() // 2 + 1).float().view(1, 1, -1, 1).sqrt()
                wgt = F.pad(wgt, [0, 0, table_q.numel() // 2, 0]) + 1e-5
                table_q = table_q.data + (table_q - table_q.data) / wgt * x.numel() / (table_q.numel() // 2 + 1)
            else:
                wgt = torch.histc(grid.data, table_q.numel()).float().view(table_q.shape).sqrt() + 1e-5
                table_q = table_q.data + (table_q - table_q.data) / wgt * x.numel() / table_q.numel()
            s = table_q.shape[2] // 2
            x_q = F.grid_sample(table_q, (F.pad(grid.data.view(1, -1, 1, 1), [1, 0]) * s).round() / s, 'nearest', 'border').view(x.shape)
            x_q = (x_q + grid - grid.data) * scale

            return x_q
        else:
            grid = (x / scale).clamp(-1, 1)
            if self.is_act:
                s = table_q.shape[2] // 2
                idx = (grid * s).round().long() + s
                x_q = table_q[0, 0, idx, 0]
            else:
                s = table_q.shape[2] - 1
                idx = ((grid + 1) / 2 * s).round().long()
                x_q = table_q[0, 0, idx, 0]
            x_q = x_q * scale

            return x_q

    def forward(self, x):
        if bool(self.scale == 0):
            if self.is_act:
                self.scale.data = (x.std() * 3).log()
            else:
                self.scale.data = (x.std() * 3).log()
        scale = self.scale.exp()

        if self.training:
            # quantize lookup table
            table_q = self._gen_table()

            # lookup
            x_q = self._lookup(x, table_q, scale)

        else:
            # lookup
            x_q = self._lookup(x, self.table_q, scale)

        return x_q


class QuantConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=None, bias=False, w_bits=4, a_bits=4):
        super(QuantConv2d, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.granu = 9

        self.weight = nn.Parameter(torch.randn(channels_out, channels_in, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels_out))
        else:
            self.bias = None

        self.act_quant = quant_lookup(channels_out, self.granu, a_bits, True) if self.a_bits < 32 else None
        self.kernel_quant = quant_lookup(channels_in * channels_out * kernel_size * kernel_size, self.granu, w_bits, False) if self.w_bits < 32 else None

    def _quantization(self):
        # activation quantization
        table_q = self.act_quant._gen_table()
        self.act_quant.register_buffer('table_q', table_q)

        # kernel quantization
        kernel_q = self.kernel_quant(self.weight) if self.w_bits < 32 else self.weight
        self.register_buffer('kernel_q', kernel_q)

    def forward(self, x):
        if self.training:
            x_q = self.act_quant(x) if self.a_bits < 32 else x
            kernel_q = self.kernel_quant(self.weight) if self.w_bits < 32 else self.weight

            out_q = F.conv2d(x_q, kernel_q, self.bias, stride=self.stride, padding=self.padding)

            return out_q
        else:
            x_q = self.act_quant(x) if self.a_bits < 32 else x

            out_q = F.conv2d(x_q, self.kernel_q, self.bias, stride=self.stride, padding=self.padding)

            return out_q


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
