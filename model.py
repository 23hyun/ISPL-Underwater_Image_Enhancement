import math
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################
## Luminance branch
## Luminance - Luminance Feature Extractor
############################################################
class Luminance(nn.Module):

    def __init__(self, sharp_strength=0.7, eps=1e-6, learnable_blur=True):
        super().__init__()

        self.eps = eps

       
        # 1. Learnable RGB -> Y
        self.rgb_to_y = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            bias=False
        )

        with torch.no_grad():
            weight = torch.tensor(
                [0.299, 0.587, 0.114],
                dtype=torch.float32
            ).view(1, 3, 1, 1)

            self.rgb_to_y.weight.copy_(weight)

        # 2. Gaussian-initialized smoothing kernel
        self.blur = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            padding=2,
            bias=False,
            padding_mode="reflect"
        )

        kernel = torch.tensor(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=torch.float32
        )

        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 5, 5)

        with torch.no_grad():
            self.blur.weight.copy_(kernel)

        if not learnable_blur:
            for p in self.blur.parameters():
                p.requires_grad = False

       
        # 3. Learnable sharpness strength
        sharp_strength = float(sharp_strength)
        sharp_strength = max(1e-4, min(sharp_strength, 2.0 - 1e-4))

        init_logit = math.log(sharp_strength / (2.0 - sharp_strength))
        self.sharp_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def forward(self, x):
        # 1. RGB -> Y
        y = self.rgb_to_y(x)  # [B, 1, H, W]

        # 2. Min-max normalization
        y_min = y.amin(dim=(2, 3), keepdim=True)
        y_max = y.amax(dim=(2, 3), keepdim=True)

        y_norm = (y - y_min) / (y_max - y_min + self.eps)

        # 3. Unsharp masking
        blur = self.blur(y_norm)
        detail = y_norm - blur

        alpha = 2.0 * torch.sigmoid(self.sharp_logit)

        y_sharp = y_norm + alpha * detail

        # 강조된 1채널 luminance map 그대로 반환
        out = torch.clamp(y_sharp, 0.0, 1.0)

        return out

class LuminanceFeatureExtractor(nn.Module):

    def __init__(self, out_channels=32):
        super().__init__()

        # 1채널 Y 맵을 feature dimension으로 투영
        self.stem = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # 3x3 16->32
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.PReLU(out_channels),
        )

        # 5x5 16->32
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                bias=True
            ),
            nn.PReLU(out_channels),
        )
        
        # 7x7 16->32
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=7,
                padding=3,
                bias=True
            ),
            nn.PReLU(out_channels),
        )
    

        # 3 5 7 특징 결합 96->32
        self.fuse = nn.Conv2d(
            in_channels=out_channels * 3,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

    def forward(self, x):
        identity = self.stem(x)

        branch1 = self.branch1(identity)
        branch2 = self.branch2(identity)
        branch3 = self.branch3(identity)

        fused = torch.cat([branch1, branch2, branch3], dim=1)

        fused = self.fuse(fused)

        out = identity + fused

        return out

    
############################################################
## SortUnsort branch
## SortUnsort x2 - SortUnsort Feature Extractor
############################################################
class SortUnsortB(nn.Module):

    def __init__(self, ref_channel=1, method="arithmetic"):
        super().__init__()
        self.method = method
        self.ref_channel = ref_channel
        self.eps = 1e-6

    def forward(self, feat):
        B, C, H, W = feat.shape
        feat_out = torch.zeros_like(feat)

        g = feat[:, self.ref_channel, :, :].reshape(B, -1)
        g_sorted, _ = torch.sort(g, dim=1)

        for c in range(C):
            if c == self.ref_channel:
                feat_out[:, c, :, :] = feat[:, c, :, :]
                continue

            x = feat[:, c, :, :].reshape(B, -1)
            x_sorted, idx = torch.sort(x, dim=1)

            if self.method == "arithmetic":
                aligned = (x_sorted + g_sorted) / 2
            else:
                aligned = 2 / (1 / (x_sorted + self.eps) + 1 / (g_sorted + self.eps))

            unsort_idx = torch.argsort(idx, dim=1)
            x_restored = torch.gather(
                aligned,
                dim=1,
                index=unsort_idx
            ).reshape(B, H, W)

            feat_out[:, c, :, :] = x_restored

        return torch.clamp(feat_out, 0, 1)

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden_channels = max(
            channels // reduction,
            4
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.attention = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=True
            ),
            nn.PReLU(hidden_channels),

            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.pool(x)
        weight = self.attention(weight)

        return x * weight

class SortUnsortFeatureExtractor(nn.Module):

    def __init__(self, out_channels=32):
        super().__init__()

        # RGB 입력을 feature dimension으로 변환
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.PReLU(out_channels)
        )

        # 공간적 특징 추출
        self.spatial = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True
        )
    

        # feature channel 간 혼합
        self.channel_mixer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )

        self.channel_attention = ChannelAttention(
            channels=out_channels,
            reduction=8
        )

    def forward(self, x):
        identity = self.stem(x)

        feature = self.spatial(identity)
        feature = self.channel_mixer(feature)
        feature = self.channel_attention(feature)

        out = identity + feature

        return out


############################################################
## ConvB2 Block
## (3x3conv-IN-PReLU) x2
############################################################
class ConvB2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(num_parameters=out_channels),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(num_parameters=out_channels),
        )

    def forward(self, x):
        return self.block(x)


############################################################
## Down
## 3x3conv-PixelUnshuffle
############################################################
class PixelDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert out_channels % 4 == 0, \
            "out_channels must be divisible by 4 for PixelUnshuffle"

        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 4,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True
            ),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.down(x)


############################################################
## ConcatFuse Block
## concat - ConvB2
############################################################
class ConcatFuse(nn.Module):

    def __init__(self, main_channels, guide_channels, out_channels):
        super().__init__()

        self.fuse = ConvB2(
            in_channels=main_channels + guide_channels,
            out_channels=out_channels
        )

    def forward(self, main, guide):
        if main.shape[-2:] != guide.shape[-2:]:
            guide = F.interpolate(
                guide,
                size=main.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([main, guide], dim=1)
        x = self.fuse(x)

        return x


############################################################
## Down
## 3x3conv-PixelUnshuffle
############################################################
class UpBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # PixelShuffle 후 channel을 in_channels로 유지
        self.up = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=True
            ),
            nn.PixelShuffle(2)
        )

        self.refine = ConvB2(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.refine(x)

        return x


############################################################
## Encoder
## 32-64-128-256-512-256
############################################################
class DualEncoderFusion(nn.Module):
    
    def __init__(self, base_channels=32):
        super().__init__()

        c1 = base_channels          # 32
        c2 = base_channels * 2      # 64
        c3 = base_channels * 4      # 128
        c4 = base_channels * 8      # 256

        # Stage 1: 32 + 32 -> 64
        self.lum_refine1 = ConvB2(c1, c1)
        self.sun_refine1 = ConvB2(c1, c1)

        self.fuse1 = ConcatFuse(
            main_channels=c1,
            guide_channels=c1,
            out_channels=c2
        )

        # main은 fuse1에서 이미 64가 되었으므로 64 유지
        self.lum_down1 = PixelDown(
            in_channels=c2,
            out_channels=c2
        )

        # guide는 32 -> 64
        self.sun_down1 = PixelDown(
            in_channels=c1,
            out_channels=c2
        )

        # Stage 2: 64 + 64 -> 128
        self.lum_refine2 = ConvB2(c2, c2)
        self.sun_refine2 = ConvB2(c2, c2)

        self.fuse2 = ConcatFuse(
            main_channels=c2,
            guide_channels=c2,
            out_channels=c3
        )

        # main은 fuse2에서 이미 128이 되었으므로 128 유지
        self.lum_down2 = PixelDown(
            in_channels=c3,
            out_channels=c3
        )

        # guide는 64 -> 128
        self.sun_down2 = PixelDown(
            in_channels=c2,
            out_channels=c3
        )

        # Stage 3: 128 + 128 -> 256
        self.lum_refine3 = ConvB2(c3, c3)
        self.sun_refine3 = ConvB2(c3, c3)

        self.fuse3 = ConcatFuse(
            main_channels=c3,
            guide_channels=c3,
            out_channels=c4
        )

        # main은 fuse3에서 이미 256이 되었으므로 256 유지
        self.lum_down3 = PixelDown(
            in_channels=c4,
            out_channels=c4
        )

        # guide는 128 -> 256
        self.sun_down3 = PixelDown(
            in_channels=c3,
            out_channels=c4
        )

        # Bottleneck: 256 + 256 -> 256
        self.lum_bottleneck = ConvB2(c4, c4)
        self.sun_bottleneck = ConvB2(c4, c4)

        self.fuse_b = ConcatFuse(
            main_channels=c4,
            guide_channels=c4,
            out_channels=c4
        )

    def forward(self, main, guide):

        # Stage 1
        l1 = self.lum_refine1(main)
        s1 = self.sun_refine1(guide)

        f1 = self.fuse1(l1, s1)       # 32 + 32 -> 64

        main = self.lum_down1(f1)    # 64 -> 64, H/2
        guide = self.sun_down1(s1)  # 32 -> 64, H/2

        # Stage 2
        l2 = self.lum_refine2(main)
        s2 = self.sun_refine2(guide)

        f2 = self.fuse2(l2, s2)       # 64 + 64 -> 128

        main = self.lum_down2(f2)    # 128 -> 128, H/4
        guide = self.sun_down2(s2)  # 64 -> 128, H/4

        # Stage 3
        l3 = self.lum_refine3(main)
        s3 = self.sun_refine3(guide)

        f3 = self.fuse3(l3, s3)       # 128 + 128 -> 256

        main = self.lum_down3(f3)    # 256 -> 256, H/8
        guide = self.sun_down3(s3)  # 128 -> 256, H/8

        
        # Bottleneck
        lb = self.lum_bottleneck(main)
        sb = self.sun_bottleneck(guide)

        b = self.fuse_b(lb, sb)       # 256 + 256 -> 256

        return f1, f2, f3, b


############################################################
## Decoder
## 256-128-64-32-3
############################################################
class SingleDecoder(nn.Module):

    def __init__(self, base_channels=32):
        super().__init__()

        c1 = base_channels          # 32
        c2 = base_channels * 2      # 64
        c3 = base_channels * 4      # 128
        c4 = base_channels * 8      # 256

        self.up3 = UpBlock(
            in_channels=c4,
            skip_channels=c4,
            out_channels=c3
        )

        self.up2 = UpBlock(
            in_channels=c3,
            skip_channels=c3,
            out_channels=c2
        )

        self.up1 = UpBlock(
            in_channels=c2,
            skip_channels=c2,
            out_channels=c1
        )

    def forward(self, features):
        f1, f2, f3, b = features

        x = self.up3(b, f3)  # 256 + 256 -> 128
        x = self.up2(x, f2)  # 128 + 128 -> 64
        x = self.up1(x, f1)  # 64 + 64 -> 32

        return x


############################################################
## Main Network
############################################################
class ColorCorrection(nn.Module):

    def __init__(self, in_channels=3, feat_channels=32, use_residual=True):
        super().__init__()

        self.use_residual = use_residual
        self.ymap = Luminance()
        self.sortunsort = SortUnsortB()

        # main branch stem: RGB -> 32
        self.lum_extractor = LuminanceFeatureExtractor(
            out_channels=feat_channels
        )

        self.sun_extractor = SortUnsortFeatureExtractor(
            out_channels=feat_channels
        )

        # dual encoder + scale-wise concat fusion
        self.encoder = DualEncoderFusion(
            base_channels=feat_channels
        )

        # single decoder
        self.decoder = SingleDecoder(
            base_channels=feat_channels
        )

        # RGB recovery: 32 -> 3
        self.recover_rgb = nn.Conv2d(
            feat_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            bias=True
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # PixelUnshuffle를 3번 쓰므로 H, W를 8의 배수로 padding
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                pad=(0, pad_w, 0, pad_h),
                mode="reflect"
            )

        # ----------------------------------------------------
        # Main input 생성
        # ----------------------------------------------------
        main_x = self.ymap(x)

        # ----------------------------------------------------
        # Guide input 생성
        # ----------------------------------------------------
        guide_x = self.sortunsort(x)
        guide_x = self.sortunsort(guide_x)

        # ----------------------------------------------------
        # Stem feature extraction
        # ----------------------------------------------------
        main_feat = self.lum_extractor(main_x)    # RGB -> 32
        guide_feat = self.sun_extractor(guide_x)  # RGB -> 32

        # ----------------------------------------------------
        # Dual encoder
        # ----------------------------------------------------
        features = self.encoder(main_feat, guide_feat)

        # ----------------------------------------------------
        # Single decoder
        # ----------------------------------------------------
        dec = self.decoder(features)                # 32 channel

        # ----------------------------------------------------
        # RGB recovery
        # ----------------------------------------------------
        out = self.recover_rgb(dec)                 # 32 -> RGB

        # if self.use_residual:
        #     out = out + x

        out = torch.clamp(out, 0, 1)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


class Transmission_Net(nn.Module):
    
    def __init__(self, in_channels=3, feat_channels=32, use_residual=True):
        super().__init__()

        self.color_correction = ColorCorrection(
            in_channels=in_channels,
            feat_channels=feat_channels,
            use_residual=use_residual
        )

    def forward(self, x):
        return self.color_correction(x)

