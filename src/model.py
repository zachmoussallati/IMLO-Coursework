"""Custom pre-activation ResNet with Squeeze-and-Excitation, built from scratch.

This is the architecture file the spec cares most about. The spec forbids
imported model classes (no torchvision.models.*) and pretrained weights, so
every layer here is a primitive (nn.Conv2d / nn.BatchNorm2d / nn.Linear / SiLU
/ Sigmoid / pooling / Dropout) wired up in my own code.

Inspirations (paper-only, no copied code; my variable names and module
structure are deliberately different from torchvision/timm reference impls):
  * He et al., 2016 - "Deep Residual Learning for Image Recognition" (ResNet),
    arXiv:1512.03385. Source of the residual connection idea and the 4-stage
    [64, 128, 256, 512] channel ladder.
  * He et al., 2016 - "Identity Mappings in Deep Residual Networks", arXiv:
    1603.05027. Source of the *pre-activation* ordering (BN -> SiLU -> Conv,
    BN -> SiLU -> Conv) which gives the identity path no nonlinearities to
    fight against and trains more stably at the depth I'm using.
  * Hu, Shen, Sun, 2018 - "Squeeze-and-Excitation Networks", arXiv:1709.01507.
    The per-channel gating module I add inside each residual branch.
  * Ramachandran, Zoph, Le, 2017 - "Searching for Activation Functions",
    arXiv:1710.05941. The Swish/SiLU activation I use throughout.
  * Goyal et al., 2017 - "Accurate, Large Minibatch SGD", arXiv:1706.02677.
    Source of the zero-init-final-BN trick I tried and later reverted in
    _init_weights - it stalled learning at this data scale (see comment).
"""

from __future__ import annotations

import torch
from torch import nn


class SqueezeExcite(nn.Module):
    """Channel-wise gating learned from a global descriptor.

    Reference: Hu et al., 2018. The block squeezes spatial dims with a global
    average pool, learns a per-channel gate via a tiny FC bottleneck, and
    multiplies the original feature map by that gate.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        # why: paper default reduction=16 with a floor of 8. The bottleneck
        # adds <1% to total params so the gain is essentially free.
        bottleneck = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Linear(channels, bottleneck, bias=True)
        self.bottleneck_act = nn.SiLU(inplace=True)
        self.expand = nn.Linear(bottleneck, channels, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = feat.shape
        descriptor = self.squeeze(feat).view(b, c)
        descriptor = self.bottleneck_act(self.reduce(descriptor))
        gate = self.gate(self.expand(descriptor))
        return feat * gate.view(b, c, 1, 1)


class StochasticDepth(nn.Module):
    """Per-sample stochastic depth (drop-path) applied to a residual branch.

    why: with drop_prob p, each training sample sees the branch output
    zeroed with probability p (so the block becomes identity for that
    sample). At eval time it's a no-op. Implementation uses the standard
    inverse scaling so the expected forward output matches eval mode.

    Reference: Huang et al., 2016, "Deep Networks with Stochastic Depth",
    arXiv:1603.09382.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # why: shape (B, 1, 1, 1) so each sample is dropped independently
        # but the whole feature map per sample is dropped together.
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class PreActSEBlock(nn.Module):
    """Pre-activation residual block with an SE gate before the residual add.

    Layout:
        x  ->  BN_a -> SiLU -> Conv3x3(stride=s) -> BN_b -> SiLU
                    -> Conv3x3 -> SE -> (+ shortcut(x))

    why pre-activation: identity path stays fully linear, which empirically
    trains more stably with BN at depth and lets the residual add fall back
    cleanly to identity.

    why SE inside the residual branch (not outside): the gate modulates only
    the learned residual; the identity path passes through untouched and the
    block can still degenerate to identity if the residual branch isn't useful.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        se_reduction: int = 16,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.bn_a = nn.BatchNorm2d(in_channels)
        self.act_a = nn.SiLU(inplace=True)
        # why: bias=False on every conv that feeds a BN. BN's affine bias
        # absorbs any constant, so the conv bias would just sit unused.
        self.conv_a = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn_b = nn.BatchNorm2d(out_channels)
        self.act_b = nn.SiLU(inplace=True)
        self.conv_b = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.gate = SqueezeExcite(out_channels, reduction=se_reduction)
        self.drop_path = StochasticDepth(drop_prob=drop_path_prob)

        # why: 1x1 projection only when shape changes (channel count or
        # stride). For "same shape" blocks the shortcut is identity and adds
        # zero params - this is the whole point of residual learning.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(feat)
        out = self.act_a(self.bn_a(feat))
        out = self.conv_a(out)
        out = self.act_b(self.bn_b(out))
        out = self.conv_b(out)
        out = self.gate(out)
        out = self.drop_path(out)
        return out + residual


class PetClassifier(nn.Module):
    """37-class image classifier for the Oxford-IIIT Pet dataset.

    Macro-architecture (input 224x224):
        stem_conv (3x3, stride 2)            -> 64 ch, 112x112
        stage1 (2 PreActSEBlocks, stride 1)  -> 64 ch, 112x112
        stage2 (2 PreActSEBlocks, stride 2)  -> 128 ch, 56x56
        stage3 (2 PreActSEBlocks, stride 2)  -> 256 ch, 28x28
        stage4 (2 PreActSEBlocks, stride 2)  -> 512 ch, 14x14
        final_bn -> SiLU -> AdaptiveAvgPool(1) -> Dropout -> Linear(512, 37)

    Total ~11.3M parameters, ~45MB on disk at fp32 (under my 50MB target).
    """

    def __init__(
        self,
        num_classes: int = 37,
        head_dropout: float = 0.2,
        se_reduction: int = 16,
        widths: tuple[int, int, int, int] = (64, 128, 256, 512),
        use_maxpool: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # why: stochastic-depth schedule. drop_path_rate is the linear-max
        # rate applied to the last block; earlier blocks scale linearly.
        # Default 0.0 reproduces the locked baseline bit-identically.
        self.drop_path_rate = drop_path_rate

        # why: `widths` is a non-breaking knob added for experimentation.
        # The default (64, 128, 256, 512) reproduces the locked baseline
        # architecture bit-identically; smaller variants (e.g. 32/64/128/
        # 256) are tried under experiments/ without touching the locked
        # recipe.
        w1, w2, w3, w4 = widths

        # why: `use_maxpool` is a second non-breaking knob. Default False
        # keeps the locked behaviour. When True, a 3x3 stride-2 MaxPool is
        # inserted right after the stem conv (ImageNet-ResNet style), so
        # stage 1 sees 56x56 features instead of 112x112. Used by
        # experiments/exp_ablation_maxpool.py to test whether the
        # ImageNet-style downsample improves things at this data scale.
        self.use_maxpool = use_maxpool

        # why: 3x3 stride-2 stem instead of the classic 7x7. With 224x224
        # inputs and only ~3.3K training images, the 7x7 ImageNet stem
        # discards too much spatial information up front. A 3x3 stride-2
        # stem halves resolution once but keeps fine-grained edges that
        # later SE blocks can weight.
        self.stem_conv = nn.Conv2d(
            in_channels=3, out_channels=w1, kernel_size=3,
            stride=2, padding=1, bias=False,
        )
        if use_maxpool:
            self.stem_maxpool: nn.Module = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1,
            )
        else:
            self.stem_maxpool = nn.Identity()
        # why: locked baseline keeps `stem_maxpool = Identity` (no MaxPool
        # after the stem). Standard ImageNet ResNets follow the 7x7 stem
        # with MaxPool(3, stride=2) to drop to 56x56 before stage 1. I
        # deliberately keep 112x112 going into stage 1 so the early SE
        # block sees high-resolution features - breeds in this dataset
        # often differ in fine markings, not gross silhouette.

        # why: linear schedule of per-block drop_path_prob across the 8
        # residual blocks. Block 0 gets ~0, block 7 gets drop_path_rate.
        num_blocks_total = 8
        drop_rates = [
            drop_path_rate * i / max(num_blocks_total - 1, 1)
            for i in range(num_blocks_total)
        ]

        self.stage1 = self._make_stage(
            in_channels=w1, out_channels=w1, num_blocks=2,
            first_stride=1, se_reduction=se_reduction,
            drop_path_probs=drop_rates[0:2],
        )
        self.stage2 = self._make_stage(
            in_channels=w1, out_channels=w2, num_blocks=2,
            first_stride=2, se_reduction=se_reduction,
            drop_path_probs=drop_rates[2:4],
        )
        self.stage3 = self._make_stage(
            in_channels=w2, out_channels=w3, num_blocks=2,
            first_stride=2, se_reduction=se_reduction,
            drop_path_probs=drop_rates[4:6],
        )
        self.stage4 = self._make_stage(
            in_channels=w3, out_channels=w4, num_blocks=2,
            first_stride=2, se_reduction=se_reduction,
            drop_path_probs=drop_rates[6:8],
        )

        # why: final BN+SiLU before pooling. Pre-activation blocks end with
        # a conv (not an activation), so without this the head would average
        # un-normalized, un-activated features.
        self.final_bn = nn.BatchNorm2d(w4)
        self.final_act = nn.SiLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # why: light dropout (p=0.2) on the pooled features. Heavier dropout
        # interacts poorly with BN; this much complements weight decay
        # without hurting convergence in 30 epochs.
        self.head_dropout = nn.Dropout(p=head_dropout)
        self.classifier = nn.Linear(w4, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
        se_reduction: int,
        drop_path_probs: list[float] | None = None,
    ) -> nn.Sequential:
        """Stack `num_blocks` pre-act SE blocks; first block does any downsample.

        why drop_path_probs: per-block stochastic-depth rates from the
        top-level linear schedule. None or [0,0,...] preserves baseline.
        """
        if drop_path_probs is None:
            drop_path_probs = [0.0] * num_blocks
        blocks: list[nn.Module] = []
        for block_idx in range(num_blocks):
            block_in = in_channels if block_idx == 0 else out_channels
            block_stride = first_stride if block_idx == 0 else 1
            blocks.append(
                PreActSEBlock(
                    in_channels=block_in,
                    out_channels=out_channels,
                    stride=block_stride,
                    se_reduction=se_reduction,
                    drop_path_prob=drop_path_probs[block_idx],
                )
            )
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        # why: He / Kaiming init matches the rectifier family. SiLU's
        # effective gain is close enough to ReLU's that nonlinearity='relu'
        # is a fine approximation; using fan_out keeps activation variance
        # roughly constant across layers.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # why: small normal init for SE FCs and the classifier.
                # Default Kaiming-uniform was producing a noisier softmax at
                # epoch 1 in early checks; std=0.01 settles faster.
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # why: I tried zeroing bn_b's gamma in every residual block (the
        # Goyal et al. 2017 trick) so each residual branch starts at zero
        # and the network is exactly identity at init. On ImageNet-scale
        # training that stabilises the early high-LR phase, but at this data
        # scale (3.3K images, 30 epochs, ~750 optimiser steps) the residual
        # branches never woke up - clean-train accuracy stalled in the low
        # double digits. Letting bn_b.weight stay at the default 1.0 puts
        # signal through the residual branches from step 1.

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.stem_conv(image)
        feat = self.stem_maxpool(feat)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.final_act(self.final_bn(feat))
        pooled = self.global_pool(feat).flatten(1)
        pooled = self.head_dropout(pooled)
        return self.classifier(pooled)


def build_model(
    num_classes: int = 37,
    widths: tuple[int, int, int, int] = (64, 128, 256, 512),
    use_maxpool: bool = False,
    drop_path_rate: float = 0.0,
) -> PetClassifier:
    """Factory used by train.py and test.py - keeps the construction in one place.

    why widths kwarg: lets experiments under experiments/ try smaller / wider
    architectures without copying the whole module. Default reproduces the
    locked baseline.

    why use_maxpool kwarg: lets the +maxpool ablation under experiments/
    swap in an ImageNet-style stem MaxPool without copying the architecture.
    Default False keeps the locked baseline behaviour.
    """
    return PetClassifier(
        num_classes=num_classes,
        widths=widths,
        use_maxpool=use_maxpool,
        drop_path_rate=drop_path_rate,
    )
