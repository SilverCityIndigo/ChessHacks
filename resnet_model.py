import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


def encoding_to_planes(encoding_tensor: torch.Tensor) -> torch.Tensor:
    """Convert flat encoding (batch, features) to 2D planes (batch, C, 8, 8).
    Assumes first 12*64 values are piece planes (standard 12: P,N,B,R,Q,K for white & black).
    Remaining features are global scalars broadcast across an 8x8 plane.
    """
    batch_size = encoding_tensor.size(0)
    feature_dim = encoding_tensor.size(1)
    piece_plane_size = 12 * 64
    if feature_dim < piece_plane_size:
        raise ValueError(f"Encoding too small ({feature_dim}) for piece planes expectation.")
    piece_flat = encoding_tensor[:, :piece_plane_size]
    piece_planes = piece_flat.view(batch_size, 12, 8, 8)
    extras = encoding_tensor[:, piece_plane_size:]
    if extras.numel() == 0:
        return piece_planes
    # Broadcast each extra feature as a constant plane
    extra_planes = extras.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 8, 8)
    return torch.cat([piece_planes, extra_planes], dim=1)


class ResNetChessNetwork(nn.Module):
    """Residual convolutional network for chess move + value prediction.
    Converts existing flat encoding into planes; extras become constant planes.
    """
    def __init__(self, num_moves: int = 4672, channels: int = 64, num_blocks: int = 6):
        super().__init__()
        self.num_moves = num_moves
        self.channels = channels
        self.num_blocks = num_blocks

        # We will dynamically infer input channels from encoding inside forward.
        # Initial projection will be created lazily.
        self.input_conv = None  # set after first forward when channel count known
        self.res_blocks = nn.ModuleList()
        self._res_initialized = False

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.policy_head = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_moves)
        )
        self.value_head = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # value in [0,1]
        )

    def _init_layers(self, in_channels: int):
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(self.channels) for _ in range(self.num_blocks)])
        self._res_initialized = True

    def forward(self, x_flat: torch.Tensor):
        # Convert flat encoding to planes
        planes = encoding_to_planes(x_flat)
        in_channels = planes.size(1)
        if not self._res_initialized:
            self._init_layers(in_channels)
        out = self.input_conv(planes)
        for block in self.res_blocks:
            out = block(out)
        pooled = self.global_pool(out).view(out.size(0), -1)
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        return policy_logits, value

    # For compatibility with existing training loop expecting feature_layers + heads
    def feature_layers(self, x_flat: torch.Tensor):
        planes = encoding_to_planes(x_flat)
        in_channels = planes.size(1)
        if not self._res_initialized:
            self._init_layers(in_channels)
        out = self.input_conv(planes)
        for block in self.res_blocks:
            out = block(out)
        pooled = self.global_pool(out).view(out.size(0), -1)
        return pooled

