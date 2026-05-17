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


class BlurPool2d(nn.Module):
    """Anti-aliased low-pass blur + subsample for shift-equivariant downsample.

    Reference: Zhang 2019, "Making Convolutional Networks Shift-Invariant
    Again", arXiv:1904.11486. Standard stride-2 convs alias the signal:
    they sample every other pixel without first applying a low-pass
    filter, so a small input shift can pivot which pixels fall on the
    sample grid. BlurPool inserts a separable binomial blur kernel
    (here `[1, 2, 1]^T * [1, 2, 1] / 16` for k=3) before the stride-2
    subsample so the downsampler sees a smoothed signal.

    Used in this codebase as a wrapper: the strided 3x3 conv in
    PreActSEBlock becomes stride-1 conv -> BlurPool2d(stride=2). Param
    cost: zero (kernel is registered as a non-trainable buffer).
    """

    def __init__(self, channels: int, stride: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        # why: separable binomial kernel from row [1,2,1]/4 (k=3). Outer
        # product gives the 2D 3x3 kernel; replicated per channel for a
        # depthwise grouped conv.
        if kernel_size == 3:
            row = torch.tensor([1.0, 2.0, 1.0])
        elif kernel_size == 5:
            row = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            raise ValueError(f"BlurPool kernel_size must be 3 or 5, got {kernel_size}")
        kernel = row[:, None] * row[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size).contiguous()
        # why: register_buffer so the kernel travels with state_dict (and
        # follows model.to(device)) without being a trainable parameter.
        self.register_buffer("kernel", kernel)
        self.channels = channels
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # why: groups=channels makes this a depthwise conv - each channel
        # is blurred independently, no cross-channel mixing.
        return torch.nn.functional.conv2d(
            x, self.kernel, stride=self.stride,
            padding=self.padding, groups=self.channels,
        )


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


class BottleneckSEBlock(nn.Module):
    """Pre-activation *bottleneck* residual block (ResNet-50 style) with SE.

    Layout:
        x -> BN_a -> SiLU -> Conv1x1(C_in -> C_mid)
           -> BN_b -> SiLU -> Conv3x3(C_mid -> C_mid, stride=s)
           -> BN_c -> SiLU -> Conv1x1(C_mid -> C_out)
           -> SE -> (+ shortcut)

    where C_mid = C_out / reduction_ratio (4 by default, ResNet-50's
    standard). The 1x1 -> 3x3 -> 1x1 sandwich does most of the
    representational work in C_mid (cheaper) channels and projects back
    to C_out at the end. A bottleneck block is ~3-4x cheaper than a
    basic block at the same output width, so the budget can buy more
    blocks instead.

    why pre-activation: same reasoning as PreActSEBlock — identity path
    stays linear, trains more stably at depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        se_reduction: int = 16,
        bottleneck_ratio: int = 4,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        # why: bottleneck width = out_channels / 4. Standard ResNet-50
        # choice; gives ~3x param reduction vs the basic 3x3-3x3 block
        # while keeping the expressivity that matters most (the 3x3).
        mid = max(out_channels // bottleneck_ratio, 8)

        self.bn_a = nn.BatchNorm2d(in_channels)
        self.act_a = nn.SiLU(inplace=True)
        # why: 1x1 compress to bottleneck width. Cheap channel mix.
        self.conv_a = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)

        self.bn_b = nn.BatchNorm2d(mid)
        self.act_b = nn.SiLU(inplace=True)
        # why: 3x3 stride-s in the middle — spatial work happens here at
        # the reduced channel count.
        self.conv_b = nn.Conv2d(
            mid, mid, kernel_size=3, stride=stride, padding=1, bias=False,
        )

        self.bn_c = nn.BatchNorm2d(mid)
        self.act_c = nn.SiLU(inplace=True)
        # why: 1x1 expand back to out_channels.
        self.conv_c = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)

        self.gate = SqueezeExcite(out_channels, reduction=se_reduction)
        self.drop_path = StochasticDepth(drop_prob=drop_path_prob)

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
        out = self.act_c(self.bn_c(out))
        out = self.conv_c(out)
        out = self.gate(out)
        out = self.drop_path(out)
        return out + residual


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
        use_blurpool: bool = False,
    ) -> None:
        super().__init__()
        self.bn_a = nn.BatchNorm2d(in_channels)
        self.act_a = nn.SiLU(inplace=True)
        # why: when BlurPool is enabled and stride > 1, the conv becomes
        # stride-1 and a separable BlurPool does the actual subsampling.
        # This restores shift-equivariance — a property a plain stride-2
        # conv breaks (Zhang 2019). For stride=1 blocks there's no
        # downsample, so BlurPool is a no-op there.
        conv_a_stride = 1 if (use_blurpool and stride > 1) else stride
        self.conv_a = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=conv_a_stride, padding=1, bias=False,
        )
        if use_blurpool and stride > 1:
            self.blurpool_a: nn.Module = BlurPool2d(out_channels, stride=stride)
        else:
            self.blurpool_a = nn.Identity()
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
        # When BlurPool is enabled and stride > 1, the projection conv is
        # also stride-1 + BlurPool, so the shortcut sees the same
        # antialiased downsample as the main path.
        if stride != 1 or in_channels != out_channels:
            sc_stride = 1 if (use_blurpool and stride > 1) else stride
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=sc_stride, bias=False,
            )
            if use_blurpool and stride > 1:
                self.shortcut_blur: nn.Module = BlurPool2d(out_channels, stride=stride)
            else:
                self.shortcut_blur = nn.Identity()
        else:
            self.shortcut = nn.Identity()
            self.shortcut_blur = nn.Identity()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut_blur(self.shortcut(feat))
        out = self.act_a(self.bn_a(feat))
        out = self.conv_a(out)
        out = self.blurpool_a(out)
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
        blocks_per_stage: tuple[int, int, int, int] = (2, 2, 2, 2),
        block_kind: str = "basic",
        use_maxpool: bool = False,
        use_blurpool: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # why: stochastic-depth schedule. drop_path_rate is the linear-max
        # rate applied to the last block; earlier blocks scale linearly.
        # Default 0.0 reproduces the locked baseline bit-identically.
        self.drop_path_rate = drop_path_rate

        # why: per-stage block counts are an experimentation knob. The
        # default (2,2,2,2) reproduces the locked 8-block layout
        # bit-identically. Larger tuples like (3,3,3,3) let an experiment
        # try a deeper net at the same widths without code edits.
        b1, b2, b3, b4 = blocks_per_stage

        # why: block_kind="basic" uses the locked PreActSEBlock (3x3 -> 3x3
        # in the main path). "bottleneck" uses BottleneckSEBlock (1x1 ->
        # 3x3 -> 1x1 with 4x channel reduction). Bottleneck blocks are
        # ~3x cheaper per block at the same output width, so an ablation
        # can stack 3x more blocks for the same param budget.
        if block_kind == "basic":
            self._block_cls: type[nn.Module] = PreActSEBlock
        elif block_kind == "bottleneck":
            self._block_cls = BottleneckSEBlock
        else:
            raise ValueError(
                f"block_kind={block_kind!r} not in {{'basic','bottleneck'}}"
            )
        # why: BlurPool is only supported by the basic block today (kept
        # the bottleneck block untouched to keep its forward simple). The
        # plumbing into _make_stage adds a use_blurpool flag through the
        # block constructor.
        self._use_blurpool = use_blurpool and block_kind == "basic"

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

        # why: linear schedule of per-block drop_path_prob across all
        # residual blocks. Block 0 gets ~0, the final block gets the full
        # drop_path_rate. Total block count depends on blocks_per_stage.
        num_blocks_total = b1 + b2 + b3 + b4
        drop_rates = [
            drop_path_rate * i / max(num_blocks_total - 1, 1)
            for i in range(num_blocks_total)
        ]
        s2_off = b1
        s3_off = b1 + b2
        s4_off = b1 + b2 + b3

        self.stage1 = self._make_stage(
            block_cls=self._block_cls,
            in_channels=w1, out_channels=w1, num_blocks=b1,
            first_stride=1, se_reduction=se_reduction,
            drop_path_probs=drop_rates[0:s2_off],
            use_blurpool=self._use_blurpool,
        )
        self.stage2 = self._make_stage(
            block_cls=self._block_cls,
            in_channels=w1, out_channels=w2, num_blocks=b2,
            first_stride=2, se_reduction=se_reduction,
            drop_path_probs=drop_rates[s2_off:s3_off],
            use_blurpool=self._use_blurpool,
        )
        self.stage3 = self._make_stage(
            block_cls=self._block_cls,
            in_channels=w2, out_channels=w3, num_blocks=b3,
            first_stride=2, se_reduction=se_reduction,
            drop_path_probs=drop_rates[s3_off:s4_off],
            use_blurpool=self._use_blurpool,
        )
        self.stage4 = self._make_stage(
            block_cls=self._block_cls,
            in_channels=w3, out_channels=w4, num_blocks=b4,
            first_stride=2, se_reduction=se_reduction,
            drop_path_probs=drop_rates[s4_off:num_blocks_total],
            use_blurpool=self._use_blurpool,
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
        block_cls: type[nn.Module],
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        first_stride: int,
        se_reduction: int,
        drop_path_probs: list[float] | None = None,
        use_blurpool: bool = False,
    ) -> nn.Sequential:
        """Stack `num_blocks` residual blocks; first block does any downsample.

        why block_cls: PreActSEBlock for "basic" (3x3 -> 3x3) or
        BottleneckSEBlock for "bottleneck" (1x1 -> 3x3 -> 1x1). Both
        have the same (in_channels, out_channels, stride, se_reduction,
        drop_path_prob) signature so this factory works for either.

        why drop_path_probs: per-block stochastic-depth rates from the
        top-level linear schedule. None or [0,0,...] preserves baseline.
        """
        if drop_path_probs is None:
            drop_path_probs = [0.0] * num_blocks
        blocks: list[nn.Module] = []
        for block_idx in range(num_blocks):
            block_in = in_channels if block_idx == 0 else out_channels
            block_stride = first_stride if block_idx == 0 else 1
            # why: only PreActSEBlock accepts use_blurpool today; pass it
            # only when supported so the bottleneck block stays unchanged.
            kw: dict = dict(
                in_channels=block_in,
                out_channels=out_channels,
                stride=block_stride,
                se_reduction=se_reduction,
                drop_path_prob=drop_path_probs[block_idx],
            )
            if block_cls is PreActSEBlock:
                kw["use_blurpool"] = use_blurpool
            blocks.append(block_cls(**kw))
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
    blocks_per_stage: tuple[int, int, int, int] = (2, 2, 2, 2),
    block_kind: str = "basic",
    se_reduction: int = 16,
    use_maxpool: bool = False,
    use_blurpool: bool = False,
    drop_path_rate: float = 0.0,
) -> PetClassifier:
    """Factory used by train.py and test.py - keeps the construction in one place.

    why widths kwarg: lets experiments under experiments/ try smaller / wider
    architectures without copying the whole module. Default reproduces the
    locked baseline.

    why blocks_per_stage kwarg: lets the +deeper ablation try 3 (or more)
    residual blocks per stage without code edits. Default (2,2,2,2)
    reproduces the locked 8-block layout bit-identically.

    why use_maxpool kwarg: lets the +maxpool ablation under experiments/
    swap in an ImageNet-style stem MaxPool without copying the architecture.
    Default False keeps the locked baseline behaviour.
    """
    return PetClassifier(
        num_classes=num_classes,
        widths=widths,
        blocks_per_stage=blocks_per_stage,
        block_kind=block_kind,
        se_reduction=se_reduction,
        use_maxpool=use_maxpool,
        use_blurpool=use_blurpool,
        drop_path_rate=drop_path_rate,
    )
