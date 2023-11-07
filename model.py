"""
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

"""

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from TDNN import TDNNBlock, BatchNorm1d


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(
            width_range[0], width_range[1], (batch, 1), device=x.device
        ).unsqueeze(2)
        mask_pos = torch.randint(
            0, max(1, D - mask_len.max()), (batch, 1), device=x.device
        ).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        C,
        ch_in=80,
        latent_dim=192,
        embed_dim=256,
        embed_reps=2,
        attn_mlp_dim=256,
        trnfr_mlp_dim=256,
        trnfr_heads=8,
        dropout=0.2,
        trnfr_layers=3,
        n_blocks=2,
        max_len=10000,
        final_layer="fc",
    ):
        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=20,
                f_max=7600,
                window_fn=torch.hamming_window,
                n_mels=80,
            ),
        )

        self.specaug = FbankAug()  # Spec augmentation

        # self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        # self.relu   = nn.ReLU()
        # self.bn1    = nn.BatchNorm1d(C)
        # self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        # self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        # self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        # self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        # self.attention = nn.Sequential(
        #     nn.Conv1d(4608, 256, kernel_size=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Tanh(), # I add this layer
        #     nn.Conv1d(256, 1536, kernel_size=1),
        #     nn.Softmax(dim=2),
        #     )
        # self.bn5 = nn.BatchNorm1d(3072)
        # self.fc6 = nn.Linear(3072, 192)
        # self.bn6 = nn.BatchNorm1d(192)

        self.ch_expansion = TDNNBlock(
            in_channels=ch_in, out_channels=embed_dim, kernel_size=1, dilation=1
        )

        # Initialize latent array
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((latent_dim, 1, embed_dim)), mean=0, std=0.02, a=-2, b=2
            )
        )

        # Initialize embedding with position encoding
        self.embed = ACANetPositionalEncoding1D(d_model=embed_dim, max_len=max_len)

        # Initialize arbitrary number of blocks
        self.ACA_blocks = nn.ModuleList(
            [
                ACABlock(
                    embed_dim=embed_dim,  # n_encoder_out
                    embed_reps=embed_reps,  # number of times to run the embedding cross attention
                    attn_mlp_dim=attn_mlp_dim,  # typical transformer MLP bottleneck dim, for the encoder
                    trnfr_mlp_dim=trnfr_mlp_dim,  # for the latent transformer
                    trnfr_heads=trnfr_heads,  # for the latent transformer
                    dropout=dropout,
                    trnfr_layers=trnfr_layers,
                )  # number of layers in each block
                for b in range(n_blocks)
            ]
        )

        # Compress embed dimension
        # final_later determines the type. currently implemented is 'fc' and '1dE' and '1dL'
        self.fl = final_layer

        if self.fl == "1dE":
            self.ch_compression = nn.Conv1d(embed_dim, 1, 1)
            self.final_norm = BatchNorm1d(input_size=latent_dim)
        elif self.fl == "1dL":
            self.ch_compression = nn.Conv1d(latent_dim, 1, 1)
            self.final_norm = BatchNorm1d(input_size=embed_dim)
        elif self.fl == "fc":
            self.ch_compression = nn.Linear(embed_dim * latent_dim, latent_dim)
            self.final_norm = BatchNorm1d(input_size=latent_dim)
        else:
            raise Exception("invalid final layer configuration")

        self.embed_reps = embed_reps

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)
        if len(x.shape) != 3:
            raise Exception("Check formatting of input")

        # Expects x to be in BATCH FIRST format [Batch, Filters, Time]
        x = x.permute(2, 1, 0)
        x = self.ch_expansion(x)  # perform channel expansion before anything else

        # First we expand our latent query matrix to size of batch
        batch_size = x.shape[0]
        input_length = x.shape[2]
        latent = self.latent.expand(-1, batch_size, -1)

        # Next, we pass the image through the embedding module to get flattened input
        x = self.embed(x)

        # Next, we permute the input x because for the ACA Blocks, x needs to be [time, batch, filters]
        x = x.permute(2, 0, 1)

        # Next, we iteratively pass the latent matrix and image embedding through
        # ACA blocks
        for pb in self.ACA_blocks:
            latent = pb(x, latent)
        # at this point the latent has dimensions: [Latent, batch, Emb]

        # two options for 1dconv:
        # 1dE has the 1dconv run over the embedding so shape has to be [Batch, Emb, latent]
        # 1dL has the 1dconv run over the Latnets so shape has to be [Batch, Latent, Emb] or
        if self.fl == "1dE":
            # [Batch, Emb, latent] Emb was originally the channel dimension anyway
            latent = latent.permute(1, 2, 0)
        elif self.fl == "1dL":
            latent = latent.permute(
                1, 0, 2
            )  ##ooops. this does not actually work because the dimensions won't be correct.
        elif self.fl == "fc":
            latent = latent.permute(
                1, 2, 0
            )  # does not matter as long as batch is put back into the first dimension
            latent = latent.flatten(1, 2)
        out = self.ch_compression(latent)
        out = self.final_norm(out.squeeze()).unsqueeze(1)
        # Finally, we project the output to the number of target classes

        return out
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.bn1(x)

        # x1 = self.layer1(x)
        # x2 = self.layer2(x+x1)
        # x3 = self.layer3(x+x1+x2)

        # x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        # x = self.relu(x)

        # t = x.size()[-1]

        # global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        # w = self.attention(global_x)

        # mu = torch.sum(x * w, dim=2)
        # sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        # x = torch.cat((mu,sg),1)
        # x = self.bn5(x)
        # x = self.fc6(x)
        # x = self.bn6(x)

        # return x


class AsymmetricCrossAttention(nn.Module):
    """Basic decoder block used both for cross-attention and the latent transformer"""

    def __init__(self, embed_dim, mlp_dim, n_heads, dropout=0.0):
        super().__init__()

        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)

        self.lnorm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, q):
        # x will be of shape [PIXELS x BATCH_SIZE x EMBED_DIM]
        # q will be of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM] when this is
        # used for cross-attention; otherwise same as x

        # attention block
        x = self.lnorm1(x)
        out, _ = self.attn(query=q, key=x, value=x)
        # out will be of shape [LATENT_DIM x BATCH_SIZE x EMBED_DIM] after matmul
        # when used for cross-attention; otherwise same as x

        # first residual connection
        resid = out + q

        # dense block
        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)

        # second residual connection
        out = out + resid

        return out


class LatentTransformer(nn.Module):
    """Latent transformer module with n_layers count of decoders."""

    def __init__(self, embed_dim, mlp_dim, n_heads, dropout, n_layers):
        super().__init__()
        self.transformer = nn.ModuleList(
            [
                AsymmetricCrossAttention(
                    embed_dim=embed_dim,
                    mlp_dim=mlp_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for l in range(n_layers)
            ]
        )
        self.ch_reduction = nn.Conv1d(embed_dim * (n_layers + 1), embed_dim, 1)

    def forward(self, l):
        L = l.clone()

        for trnfr in self.transformer:
            l = trnfr(l, l)
            L = torch.cat([L, l], 2)

        L = L.permute(0, 2, 1)
        L = torch.nn.functional.relu(self.ch_reduction(L))
        L = L.permute(0, 2, 1)

        return L


class ACABlock(nn.Module):
    """Block consisting of one cross-attention layer and one latent transformer"""

    def __init__(
        self,
        embed_dim,
        embed_reps,
        attn_mlp_dim,
        trnfr_mlp_dim,
        trnfr_heads,
        dropout,
        trnfr_layers,
    ):
        super().__init__()

        self.embed_reps = embed_reps

        self.cross_attention = nn.ModuleList(
            [
                AsymmetricCrossAttention(
                    embed_dim, attn_mlp_dim, n_heads=1, dropout=dropout
                )
                for _ in range(embed_reps)
            ]
        )

        self.latent_transformer = LatentTransformer(
            embed_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers
        )

    def forward(self, x, l):
        for ca in self.cross_attention:
            l = ca(x, l)

        l = self.latent_transformer(l)

        return l


# modified from speechbrain
class ACANetPositionalEncoding1D(nn.Module):
    """Positional encoder for the pytorch transformer.

    This was modified from the original speechbrain implementation

    Arguments
    ---------
    d_model : int
        Representation dimensionality.
    max_len : int
        Max sequence length.

    Example
    -------

    >>> x = torch.randn(5, 512, 999) #Tensor Shape [Batch, Filters, Time]
    >>> enc = ACANetPositionalEncoding1D(512)
    >>> x = enc(x)
    """

    def __init__(self, d_model, max_len):
        super(ACANetPositionalEncoding1D, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Returns the encoded output.
        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, N, L],
            where, B = Batchsize,
                   N = number of filters
                   L = time points

        NOTE: self.pe was designed originally to accept Tensor shape [B, L, N]
        However, for speechbrain, we want Tensor shape [B, N, L]. Therefore, here we must permute.
        """
        x = x.permute(0, 2, 1)
        x = x + self.pe[: x.size(0), :]
        x = x.permute(0, 2, 1)

        return x


class ACANet(nn.Module):
    """ACANet Classification Network"""

    def __init__(
        self,
        ch_in,
        latent_dim,
        embed_dim,
        embed_reps,
        attn_mlp_dim,
        trnfr_mlp_dim,
        trnfr_heads,
        dropout,
        trnfr_layers,
        n_blocks,
        max_len,
        final_layer,
    ):
        super().__init__()

        self.ch_expansion = TDNNBlock(
            in_channels=ch_in, out_channels=embed_dim, kernel_size=1, dilation=1
        )

        # Initialize latent array
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((latent_dim, 1, embed_dim)), mean=0, std=0.02, a=-2, b=2
            )
        )

        # Initialize embedding with position encoding
        self.embed = ACANetPositionalEncoding1D(d_model=embed_dim, max_len=max_len)

        # Initialize arbitrary number of blocks
        self.ACA_blocks = nn.ModuleList(
            [
                ACABlock(
                    embed_dim=embed_dim,  # n_encoder_out
                    embed_reps=embed_reps,  # number of times to run the embedding cross attention
                    attn_mlp_dim=attn_mlp_dim,  # typical transformer MLP bottleneck dim, for the encoder
                    trnfr_mlp_dim=trnfr_mlp_dim,  # for the latent transformer
                    trnfr_heads=trnfr_heads,  # for the latent transformer
                    dropout=dropout,
                    trnfr_layers=trnfr_layers,
                )  # number of layers in each block
                for b in range(n_blocks)
            ]
        )

        # Compress embed dimension
        # final_later determines the type. currently implemented is 'fc' and '1dE' and '1dL'
        self.fl = final_layer

        if self.fl == "1dE":
            self.ch_compression = nn.Conv1d(embed_dim, 1, 1)
            self.final_norm = BatchNorm1d(input_size=latent_dim)
        elif self.fl == "1dL":
            self.ch_compression = nn.Conv1d(latent_dim, 1, 1)
            self.final_norm = BatchNorm1d(input_size=embed_dim)
        elif self.fl == "fc":
            self.ch_compression = nn.Linear(embed_dim * latent_dim, latent_dim)
            self.final_norm = BatchNorm1d(input_size=latent_dim)
        else:
            raise Exception("invalid final layer configuration")

        self.embed_reps = embed_reps

    def forward(self, x):
        # x should come in as [batch, time, filters]
        if len(x.shape) != 3:
            raise Exception("Check formatting of input")

        # Expects x to be in BATCH FIRST format [Batch, Filters, Time]
        x = x.permute(0, 2, 1)
        x = self.ch_expansion(x)  # perform channel expansion before anything else

        # First we expand our latent query matrix to size of batch
        batch_size = x.shape[0]
        input_length = x.shape[2]
        latent = self.latent.expand(-1, batch_size, -1)

        # Next, we pass the image through the embedding module to get flattened input
        x = self.embed(x)

        # Next, we permute the input x because for the ACA Blocks, x needs to be [time, batch, filters]
        x = x.permute(2, 0, 1)

        # Next, we iteratively pass the latent matrix and image embedding through
        # ACA blocks
        for pb in self.ACA_blocks:
            latent = pb(x, latent)
        # at this point the latent has dimensions: [Latent, batch, Emb]

        # two options for 1dconv:
        # 1dE has the 1dconv run over the embedding so shape has to be [Batch, Emb, latent]
        # 1dL has the 1dconv run over the Latnets so shape has to be [Batch, Latent, Emb] or
        if self.fl == "1dE":
            # [Batch, Emb, latent] Emb was originally the channel dimension anyway
            latent = latent.permute(1, 2, 0)
        elif self.fl == "1dL":
            latent = latent.permute(
                1, 0, 2
            )  ##ooops. this does not actually work because the dimensions won't be correct.
        elif self.fl == "fc":
            latent = latent.permute(
                1, 2, 0
            )  # does not matter as long as batch is put back into the first dimension
            latent = latent.flatten(1, 2)
        out = self.ch_compression(latent)
        out = self.final_norm(out.squeeze()).unsqueeze(1)
        # Finally, we project the output to the number of target classes

        return out  # reorder inputs back to [Batch, filters, time] format for the rest of speechbrain


# if __name__ == "__main__":
#     per = ACANet(
#         ch_in=80,
#         latent_dim=192,
#         embed_dim=256,
#         embed_reps=2,
#         attn_mlp_dim=256,
#         trnfr_mlp_dim=256,
#         trnfr_heads=8,
#         dropout=0.2,
#         trnfr_layers=3,
#         n_blocks=2,
#         max_len=10000,
#         final_layer="fc",
#     )
#     x = torch.randn(5, 999, 80)
#     x = per(x)
#     print(x)
