import math
import torch
import torch as th
from einops import rearrange
from torch import nn
from torch.nn import functional as F
import fractions
import torchaudio


def spectro(x, n_fft, hop_length):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = th.stft(x, n_fft, hop_length, window=th.hann_window(n_fft).to(x),
                win_length=n_fft, normalized=True, center=True,
                return_complex=True, pad_mode="reflect")
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length, length):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    x = th.istft(z, n_fft, hop_length, window=th.hann_window(n_fft).to(z.real),
                 win_length=n_fft, normalized=True, length=length, center=True)
    return x.view(*other, x.shape[-1])


class LayerScale(nn.Module):
    def __init__(self, channels, init=0, channel_last=False):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.full((channels,), float(init)))

    def forward(self, x):
        return self.scale * x if self.channel_last else self.scale[:, None] * x


class DConv(nn.Module):
    def __init__(self, channels, compress=8, depth=2, init=1e-3, kernel=3):
        super().__init__()
        hidden = int(channels / compress)
        self.layers = nn.ModuleList()
        for d in range(depth):
            dilation = 2 ** d
            padding = dilation * (kernel // 2)
            self.layers.append(nn.Sequential(
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                nn.GroupNorm(1, hidden), nn.GELU(),
                nn.Conv1d(hidden, 2 * channels, 1),
                nn.GroupNorm(1, 2 * channels), nn.GLU(1),
                LayerScale(channels, init),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class ScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, scale=10.0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        weight = torch.cumsum(self.embedding.weight.data, dim=0)
        weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
        self.embedding.weight.data[:] = weight / scale
        self.scale = scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, freq=True, context=0,
                 dconv_kw={}):
        super().__init__()
        self.freq = freq
        self.stride = stride
        pad = kernel_size // 4
        klass = nn.Conv2d if freq else nn.Conv1d
        if freq:
            kernel_size, stride, pad = [kernel_size, 1], [stride, 1], [pad, 0]
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
        self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x):
        if not self.freq:
            if x.dim() == 4:
                B, C, Fr, T = x.shape
                x = x.view(B, -1, T)
            le = x.shape[-1]
            if le % self.stride:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = F.gelu(self.conv(x))
        if self.freq:
            B, C, Fr, T = y.shape
            y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y).view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = self.dconv(y)
        return F.glu(self.rewrite(y), dim=1)


class HDecLayer(nn.Module):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, freq=True,
                 context=1, dconv_kw={}):
        super().__init__()
        pad = kernel_size // 4
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        klass = nn.Conv2d if freq else nn.Conv1d
        klass_tr = nn.ConvTranspose2d if freq else nn.ConvTranspose1d
        ks, st = ([kernel_size, 1], [stride, 1]) if freq else (kernel_size, stride)
        self.conv_tr = klass_tr(chin, chout, ks, st)
        self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
        self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)
        x = x + skip
        y = F.glu(self.rewrite(x), dim=1)
        if self.freq:
            B, C, Fr, T = y.shape
            y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y).view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = self.dconv(y)
        z = self.conv_tr(y)
        if self.freq:
            z = z[..., self.pad: -self.pad, :]
        else:
            z = z[..., self.pad: self.pad + length]
        if not self.last:
            z = F.gelu(z)
        return z, y


def create_sin_embedding(length, dim, device, max_period=10000):
    pos = torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(half_dim, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def create_2d_sin_embedding(d_model, height, width, device, max_period=10000):
    pe = torch.zeros(d_model, height, width)
    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1:: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe[None, :].to(device)


class MyGroupNorm(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, activation=activation, batch_first=True,
                         norm_first=True)
        self.norm_out = MyGroupNorm(1, d_model)
        self.gamma_1 = LayerScale(d_model, 1e-4, True)
        self.gamma_2 = LayerScale(d_model, 1e-4, True)

    def forward(self, x, **unused):
        x = x + self.gamma_1(self._sa_block(self.norm1(x), None, None))
        x = x + self.gamma_2(self._ff_block(self.norm2(x)))
        return self.norm_out(x)


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_out = MyGroupNorm(1, d_model)
        self.gamma_1 = LayerScale(d_model, 1e-4, True)
        self.gamma_2 = LayerScale(d_model, 1e-4, True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, q, k):
        x = q + self.gamma_1(self.dropout1(
            self.cross_attn(self.norm1(q), self.norm2(k), self.norm2(k), need_weights=False)[0]))
        x = x + self.gamma_2(self._ff_block(self.norm3(x)))
        return self.norm_out(x)

    def _ff_block(self, x):
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, hidden_scale=4.0, num_heads=8, num_layers=5, dropout=0.0,
                 max_period=10000.0, gelu=True):
        super().__init__()
        hidden_dim = int(dim * hidden_scale)
        self.num_layers = num_layers
        self.max_period = max_period
        self.norm_in = nn.LayerNorm(dim)
        self.norm_in_t = nn.LayerNorm(dim)
        activation = F.gelu if gelu else F.relu
        self.layers = nn.ModuleList()
        self.layers_t = nn.ModuleList()
        common = dict(d_model=dim, nhead=num_heads, dim_feedforward=hidden_dim,
                      dropout=dropout, activation=activation)
        for idx in range(num_layers):
            klass = MyTransformerEncoderLayer if idx % 2 == 0 else CrossTransformerEncoderLayer
            self.layers.append(klass(**common))
            self.layers_t.append(klass(**common))

    def forward(self, x, xt):
        B, C, Fr, T1 = x.shape
        pos_2d = rearrange(create_2d_sin_embedding(C, Fr, T1, x.device, self.max_period),
                           "b c fr t1 -> b (t1 fr) c")
        x = (self.norm_in(rearrange(x, "b c fr t1 -> b (t1 fr) c")) + pos_2d).to(x.dtype)

        B, C, T2 = xt.shape
        pos = rearrange(create_sin_embedding(T2, C, x.device, self.max_period), "t2 b c -> b t2 c")
        xt = (self.norm_in_t(rearrange(xt, "b c t2 -> b t2 c")) + pos).to(xt.dtype)

        for idx in range(self.num_layers):
            if idx % 2 == 0:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = rearrange(x, "b (t1 fr) c -> b c fr t1", t1=T1)
        xt = rearrange(xt, "b t2 c -> b c t2")
        return x, xt


class HTDemucs(nn.Module):
    def __init__(self, sources=['drums', 'bass', 'other', 'vocals'], audio_channels=2, channels=48, growth=2, nfft=4096, cac=True,
                 depth=4, freq_emb=0.2, emb_scale=10, kernel_size=8, time_stride=2, stride=4,
                 context=1, context_enc=0, dconv_depth=2, dconv_comp=8, dconv_init=1e-3,
                 bottom_channels=512, t_layers=5, t_hidden_scale=4.0, t_heads=8, t_dropout=0.0,
                 t_max_period=10000.0, t_gelu=True, samplerate=44100, segment=fractions.Fraction(39, 5), **unused):
        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.samplerate = samplerate
        self.segment = segment
        self.nfft = nfft
        self.hop_length = nfft // 4

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin * 2
        chout = chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            dconv_kw = {"depth": dconv_depth, "compress": dconv_comp, "init": dconv_init}
            kw = {"kernel_size": kernel_size, "stride": stride, "dconv_kw": dconv_kw}
            kwt = dict(kw, freq=False)

            self.encoder.append(HEncLayer(chin_z, chout_z, context=context_enc, **kw))
            self.tencoder.append(HEncLayer(chin, chout, freq=False, context=context_enc, **kw))

            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin * 2
            self.decoder.insert(0, HDecLayer(chout_z, chin_z, last=index == 0, context=context, **kw))
            self.tdecoder.insert(0, HDecLayer(chout, chin, last=index == 0, context=context, **kwt))

            chin, chin_z = chout, chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            freqs //= stride
            if index == 0:
                self.freq_emb = ScaledEmbedding(freqs, chin_z, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        transformer_channels = channels * growth ** (depth - 1)
        self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
        self.channel_downsampler = nn.Conv1d(bottom_channels, transformer_channels, 1)
        self.channel_upsampler_t = nn.Conv1d(transformer_channels, bottom_channels, 1)
        self.channel_downsampler_t = nn.Conv1d(bottom_channels, transformer_channels, 1)

        self.crosstransformer = CrossTransformerEncoder(
            dim=bottom_channels, hidden_scale=t_hidden_scale, num_heads=t_heads,
            num_layers=t_layers, dropout=t_dropout, max_period=t_max_period, gelu=t_gelu)

    def _spec(self, x):
        hl, nfft = self.hop_length, self.nfft
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = F.pad(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = spectro(x.float(), nfft, hl)[..., :-1, :]
        return z[..., 2: 2 + le]

    def _ispec(self, z, length):
        hl = self.hop_length
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        return ispectro(z, hl, le)[..., pad: pad + length]

    def _magnitude(self, z):
        # cac: 复数实虚部当作额外通道
        B, C, Fr, T = z.shape
        return torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(B, C * 2, Fr, T)

    def _mask(self, m):
        # cac: m 即完整复数谱(通道形式), 还原成复数
        B, S, C, Fr, T = m.shape
        out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        return torch.view_as_complex(out.contiguous())

    def forward(self, mix):
        length = mix.shape[-1]
        z = self._spec(mix)
        x = self._magnitude(z).to(device=mix.device, dtype=mix.dtype)

        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            lengths_t.append(xt.shape[-1])
            xt = self.tencoder[idx](xt)
            saved_t.append(xt)
            x = encode(x)
            if idx == 0:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        b, c, f, t = x.shape
        x = rearrange(x, "b c f t-> b c (f t)")
        x = self.channel_upsampler(x)
        x = rearrange(x, "b c (f t)-> b c f t", f=f)
        xt = self.channel_upsampler_t(xt)

        x, xt = self.crosstransformer(x, xt)

        x = rearrange(x, "b c f t-> b c (f t)")
        x = self.channel_downsampler(x)
        x = rearrange(x, "b c (f t)-> b c f t", f=f)
        xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            x, pre = decode(x, saved.pop(-1), lengths.pop(-1))
            xt, _ = self.tdecoder[idx](xt, saved_t.pop(-1), lengths_t.pop(-1))

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T) * std[:, None] + mean[:, None]
        x = self._ispec(self._mask(x.float()), length).to(mix.dtype)

        xt = xt.view(B, S, -1, length) * stdt[:, None] + meant[:, None]
        return xt + x
    
    def extract_track(self, audio, sample_rate, track="vocals"):
        if sample_rate != 44100:
            audio = torchaudio.functional.resample(audio, sample_rate, 44100)
        ref = audio.mean(0)
        audio = (audio - ref.mean()) / (ref.std() + 1e-8)
        out = apply_model(self, audio[None], shifts=1, split=True, overlap=0.25,
                          device=audio.device, progress=True)[0]
        out = out * (ref.std() + 1e-8) + ref.mean()
        out = out / max(1.01 * out.abs().max(), 1)
        out = out.clamp_(-1, 1).cpu()
        out = out[self.sources.index(track)]
        return out

def center_trim(tensor, length):
    delta = tensor.size(-1) - length
    if delta:
        tensor = tensor[..., delta // 2: -(delta - delta // 2)]
    return tensor


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total = tensor.shape[-1]
        if isinstance(tensor, TensorChunk):
            self.tensor, self.offset = tensor.tensor, offset + tensor.offset
        else:
            self.tensor, self.offset = tensor, offset
        self.length = total - offset if length is None else min(total - offset, length)
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total = self.tensor.shape[-1]
        start = self.offset - delta // 2
        end = start + target_length
        correct_start, correct_end = max(0, start), min(total, end)
        out = F.pad(self.tensor[..., correct_start:correct_end],
                    (correct_start - start, end - correct_end))
        return out


def apply_model(model, mix, shifts=1, split=True, overlap=0.25, device=None, segment=None,
                progress=False):
    device = th.device(device) if device is not None else mix.device
    batch, channels, length = mix.shape

    if shifts:
        max_shift = int(0.5 * model.samplerate)
        mix = mix if isinstance(mix, TensorChunk) else TensorChunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0.0
        for _ in range(shifts):
            offset = int(th.randint(0, max_shift, (1,)).item())
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(model, shifted, shifts=0, split=split, overlap=overlap,
                                      device=device, segment=segment, progress=progress)
            out += shifted_out[..., max_shift - offset:]
        return out / shifts

    if split:
        out = th.zeros(batch, len(model.sources), channels, length, device=mix.device)
        sum_weight = th.zeros(length, device=mix.device)
        if segment is None:
            segment = model.segment
        segment_length = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = range(0, length, stride)
        scale = float(format(stride / model.samplerate, ".2f"))
        weight = th.cat([th.arange(1, segment_length // 2 + 1, device=device),
                         th.arange(segment_length - segment_length // 2, 0, -1, device=device)])
        weight = weight / weight.max()
        iterator = offsets
        for offset in iterator:
            chunk = TensorChunk(mix, offset, segment_length)
            chunk_out = apply_model(model, chunk, shifts=0, split=False, device=device, segment=segment)
            cl = chunk_out.shape[-1]
            out[..., offset:offset + segment_length] += (weight[:cl] * chunk_out).to(mix.device)
            sum_weight[offset:offset + segment_length] += weight[:cl].to(mix.device)
        return out / sum_weight

    valid_length = int(segment * model.samplerate)
    mix = mix if isinstance(mix, TensorChunk) else TensorChunk(mix)
    padded_mix = mix.padded(valid_length).to(device)
    out = model(padded_mix)
    return center_trim(out, length)
