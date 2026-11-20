"""Neural LLR architectures; input (batch, 2, F, 79), output (batch, 174)."""

import torch
from torch import nn

DATA_SYMBOL_POSITIONS = list(range(7, 36)) + list(range(43, 72))


def _select_data_symbols(x, idx):
    """Extract the 58 data symbol columns from the 79-symbol sequence."""
    return x.index_select(-1, idx)


class PerSymbolMLP(nn.Module):
    """Estimate soft bits independently for each data symbol."""

    def __init__(self, freq_bins=32, hidden=64):
        super().__init__()
        self.freq_bins = freq_bins
        self.register_buffer(
            "data_symbol_idx",
            torch.tensor(DATA_SYMBOL_POSITIONS, dtype=torch.long),
            persistent=False,
        )
        in_features = 2 * freq_bins
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        d = _select_data_symbols(x, self.data_symbol_idx)
        B = d.shape[0]
        d = d.permute(0, 3, 1, 2).reshape(B, 58, 2 * self.freq_bins)
        out = self.net(d)
        return out.reshape(B, 174)


class Conv1DNet(nn.Module):
    """Estimate soft bits using convolutions across symbol time."""

    def __init__(self, freq_bins=32, channels=64, kernel_size=7, depth=4):
        super().__init__()
        self.freq_bins = freq_bins
        self.register_buffer(
            "data_symbol_idx",
            torch.tensor(DATA_SYMBOL_POSITIONS, dtype=torch.long),
            persistent=False,
        )
        in_channels = 2 * freq_bins
        layers = [
            nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size // 2),
            nn.GELU(approximate="tanh"),
        ]
        for _ in range(depth - 1):
            layers += [
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.GELU(approximate="tanh"),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 3)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, 2 * self.freq_bins, 79)
        feats = self.backbone(x)
        feats = feats.permute(0, 2, 1)
        data_feats = feats.index_select(1, self.data_symbol_idx)
        out = self.head(data_feats)
        return out.reshape(B, 174)


class Conv2DNet(nn.Module):
    """Estimate soft bits using local frequency and time structure."""

    def __init__(self, freq_bins=32, channels=32, depth=4):
        super().__init__()
        self.register_buffer(
            "data_symbol_idx",
            torch.tensor(DATA_SYMBOL_POSITIONS, dtype=torch.long),
            persistent=False,
        )

        def block(cin, cout, kf=3, kt=7):
            return nn.Sequential(
                nn.Conv2d(cin, cout, (kf, kt), padding=(kf // 2, kt // 2)),
                nn.GELU(approximate="tanh"),
            )

        layers = [block(2, channels)]
        for _ in range(depth - 1):
            layers.append(block(channels, channels))
        self.backbone = nn.Sequential(*layers)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.head = nn.Linear(channels, 3)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.freq_pool(feats).squeeze(2)
        feats = feats.permute(0, 2, 1)
        data_feats = feats.index_select(1, self.data_symbol_idx)
        out = self.head(data_feats)
        return out.reshape(x.shape[0], 174)


class SymbolTransformer(nn.Module):
    """Estimate soft bits using attention across the frame."""

    def __init__(
        self,
        freq_bins=32,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.register_buffer(
            "data_symbol_idx",
            torch.tensor(DATA_SYMBOL_POSITIONS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "position_idx", torch.arange(79, dtype=torch.long), persistent=False
        )
        self.input_proj = nn.Linear(2 * freq_bins, d_model)
        self.pos_embed = nn.Embedding(79, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, 3)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, 2 * self.freq_bins, 79).permute(0, 2, 1)
        tokens = self.input_proj(x)
        tokens = tokens + self.pos_embed(self.position_idx)
        out = self.encoder(tokens)
        data_out = out.index_select(1, self.data_symbol_idx)
        llrs = self.head(data_out)
        return llrs.reshape(B, 174)


def build_model(name: str, freq_bins=32, **kwargs) -> nn.Module:
    registry = {
        "mlp": PerSymbolMLP,
        "conv1d": Conv1DNet,
        "conv2d": Conv2DNet,
        "transformer": SymbolTransformer,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name](freq_bins=freq_bins, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_NAMES = ("mlp", "conv1d", "conv2d", "transformer")
