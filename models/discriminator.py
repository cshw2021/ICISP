import torch
import torch.nn as nn
import torch.nn.functional as F


class AFFB(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(AFFB, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)  # max(12//2, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class SFTLayer(nn.Module):
    def __init__(self, cond_c, nf):
        super().__init__()

        self.conv_scale1 = nn.Conv2d(cond_c, nf, kernel_size=1)
        self.conv_scale2 = nn.Conv2d(nf, nf, kernel_size=1)

        self.conv_shift1 = nn.Conv2d(cond_c, nf, kernel_size=1)
        self.conv_shift2 = nn.Conv2d(nf, nf, kernel_size=1)

        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, condition):
        scale = self.conv_scale2(self.act(self.conv_scale1(condition)))
        shift = self.conv_shift2(self.act(self.conv_shift1(condition)))
        _, x_c, _, _ = x.size()
        out = torch.cat([x[:, :x_c // 2, ...], scale * x[:, x_c // 2:, ...] + shift], dim=1)

        return out


class DynSFTLayer(nn.Module):
    '''
    DSFT
    '''
    def __init__(self, cond_c, nf):
        super().__init__()

        self.conv_scale1 = nn.Conv2d(cond_c, int(nf / 2), kernel_size=1)
        self.conv_scale2 = nn.Conv2d(int(nf / 2), int(nf / 2), kernel_size=1)

        self.conv_shift1 = nn.Conv2d(cond_c, int(nf / 2), kernel_size=1)
        self.conv_shift2 = nn.Conv2d(int(nf / 2), int(nf / 2), kernel_size=1)

        self.conv_scale11 = nn.Conv2d(cond_c, int(nf * 2 / 3), kernel_size=1)
        self.conv_scale22 = nn.Conv2d(int(nf * 2 / 3), int(nf * 2 / 3), kernel_size=1)

        self.conv_shift11 = nn.Conv2d(cond_c, int(nf * 2 / 3), kernel_size=1)
        self.conv_shift22 = nn.Conv2d(int(nf * 2 / 3), int(nf * 2 / 3), kernel_size=1)

        self.conv_scale111 = nn.Conv2d(cond_c, int(nf * 3 / 4), kernel_size=1)
        self.conv_scale222 = nn.Conv2d(int(nf * 3 / 4), int(nf * 3 / 4), kernel_size=1)

        self.conv_shift111 = nn.Conv2d(cond_c, int(nf * 3 / 4), kernel_size=1)
        self.conv_shift222 = nn.Conv2d(int(nf * 3 / 4), int(nf * 3 / 4), kernel_size=1)

        self.conv_scale1111 = nn.Conv2d(cond_c, int(nf * 4 / 5), kernel_size=1)
        self.conv_scale2222 = nn.Conv2d(int(nf * 4 / 5), int(nf * 4 / 5), kernel_size=1)

        self.conv_shift1111 = nn.Conv2d(cond_c, int(nf * 4 / 5), kernel_size=1)
        self.conv_shift2222 = nn.Conv2d(int(nf * 4 / 5), int(nf * 4 / 5), kernel_size=1)

        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.dynweight1 = torch.nn.Parameter(torch.Tensor([0.2]), requires_grad=True)
        self.dynweight2 = torch.nn.Parameter(torch.Tensor([0.2]), requires_grad=True)
        self.dynweight3 = torch.nn.Parameter(torch.Tensor([0.2]), requires_grad=True)
        self.dynweight4 = torch.nn.Parameter(torch.Tensor([0.2]), requires_grad=True)

        self.out_func = nn.Conv2d(nf, nf, kernel_size=1)

    def forward(self, x, condition):
        _, x_c, _, _ = x.size()
        scale1 = self.conv_scale2(self.act(self.conv_scale1(condition)))
        shift1 = self.conv_shift2(self.act(self.conv_shift1(condition)))
        out1 = torch.cat([x[:, int(x_c / 2):, ...], scale1 * x[:, :int(x_c / 2), ...] + shift1], dim=1)

        scale11 = self.conv_scale22(self.act(self.conv_scale11(condition)))
        shift11 = self.conv_shift22(self.act(self.conv_shift11(condition)))
        out2 = torch.cat([x[:, int(x_c * 2 / 3):, ...], scale11 * x[:, :int(x_c * 2 / 3), ...] + shift11], dim=1)

        scale111 = self.conv_scale222(self.act(self.conv_scale111(condition)))
        shift111 = self.conv_shift222(self.act(self.conv_shift111(condition)))
        out3 = torch.cat([x[:, int(x_c * 3 / 4):, ...], scale111 * x[:, :int(x_c * 3 / 4), ...] + shift111],
                         dim=1)

        scale1111 = self.conv_scale2222(self.act(self.conv_scale1111(condition)))
        shift1111 = self.conv_shift2222(self.act(self.conv_shift1111(condition)))
        out4 = torch.cat([x[:, int(x_c * 4 / 5):, ...], scale1111 * x[:, :int(x_c * 4 / 5), ...] + shift1111],
                         dim=1)

        out = self.out_func(
            out1 * self.dynweight1 + out2 * self.dynweight2 + out3 * self.dynweight3 + out4 * self.dynweight4)

        return out

class DINOv2(nn.Module):
    """Use DINOv2 pre-trained models
    """

    def __init__(self, version='large', freeze=False, load_from=None):
        super().__init__()

        if version == 'large':
            self.dinov2 = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_vitl14', source='local',
                                         pretrained=True)
        else:
            raise NotImplementedError

        if load_from is not None:
            d = torch.load(load_from, map_location='cpu')
            new_d = {}
            for key, value in d.items():
                if 'pretrained' in key:
                    new_d[key.replace('pretrained.', '')] = value
            self.dinov2.load_state_dict(new_d)

        self.freeze = freeze

    def forward(self, inputs):
        B, _, h, w = inputs.shape

        if self.freeze:
            with torch.no_grad():
                features = self.dinov2.get_intermediate_layers(inputs, 1)  # 4
        else:
            features = self.dinov2.get_intermediate_layers(inputs, 1)

        outs = []
        for feature in features:
            C = feature.shape[-1]
            feature = feature.permute(0, 2, 1).reshape(B, C, h // 14, w // 14).contiguous()
            outs.append(feature)
        out = torch.cat(outs, dim=1)

        return out

class Discriminator(nn.Module):
    def __init__(self, context_dims=320, spectral_norm=True):
        """
        Convolutional patchGAN discriminator used in [1].
        Accepts as input generator output G(z) or x ~ p*(x) where
        p*(x) is the true data distribution.
        Contextual information provided is encoder output y = E(x)
        ========
        Arguments:
        image_dims:     Dimensions of input image, (C_in,H,W)
        context_dims:   Dimensions of contextual information, (C_in', H', W')
        C:              Bottleneck depth, controls bits-per-pixel
                        C = 220 used in [1], C = C_in' if encoder output used
                        as context.

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression",
            arXiv:2006.09965 (2020).
        """
        super(Discriminator, self).__init__()

        self.context_dims = context_dims
        im_channels = 3
        kernel_dim = 4
        context_C_out = 12
        filters = (64, 128, 256, 512)
        self.dino = DINOv2(freeze=True).cuda()
        # Upscale encoder output - (C, 16, 16) -> (12, 256, 256)
        self.context_conv = nn.Conv2d(context_dims, context_C_out, kernel_size=3, padding=1, padding_mode='reflect')
        self.context_conv_z = nn.Conv2d(1024, context_C_out, kernel_size=3, padding=1, padding_mode='reflect')

        # Layer / normalization options
        # TODO: calculate padding properly
        cnn_kwargs = dict(stride=2, padding=1, padding_mode='reflect')
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        if spectral_norm is True:
            norm = nn.utils.spectral_norm
        else:
            norm = nn.utils.weight_norm

        # (C_in + C_in', 256,256) -> (64,128,128), with implicit padding
        # TODO: Check if removing spectral norm in first layer works
        self.conv1 = norm(nn.Conv2d(im_channels, filters[0], kernel_dim, **cnn_kwargs))

        # (128,128) -> (64,64)
        self.conv2 = norm(nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs))

        # (64,64) -> (32,32)
        self.conv3 = norm(nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs))

        # (32,32) -> (16,16)
        self.conv4 = norm(nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs))

        self.conv_out = nn.Conv2d(filters[3], 1, kernel_size=1, stride=1)

        self.skff = AFFB(context_C_out, 2, reduction=2)

        self.up1 = nn.Upsample(scale_factor=8, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.SFT1 = DynSFTLayer(context_C_out, filters[0])
        self.SFT2 = DynSFTLayer(context_C_out, filters[1])
        self.SFT3 = DynSFTLayer(context_C_out, filters[2])

    def forward(self, x, y, z):
        """
        x: Concatenated real/gen images
        y: Quantized latents
        z: GT
        """
        z = F.interpolate(z, size=(224, 224), mode='bilinear')
        z = self.dino(z)
        z = self.activation(self.context_conv_z(z))
        y = self.activation(self.context_conv(y))
        # print(z.size())
        # print(y.size())
        condition_fuse = self.skff([z, y])  # (12) 16x16

        x = self.activation(self.conv1(x))  # (3->64) 256->128

        condition = self.up1(condition_fuse)  # (12) 16->128
        x = self.SFT1(x, condition)
        x = self.activation(self.conv2(x))  # (64->128) 128->64

        condition = self.up2(condition_fuse)  # (12) 16->64
        x = self.SFT2(x, condition)
        x = self.activation(self.conv3(x))  # (128->256) 64->32

        condition = self.up3(condition_fuse)  # (12) 16->32
        x = self.SFT3(x, condition)
        x = self.activation(self.conv4(x))  # (256->512) 32->16

        out = self.conv_out(x).view(-1, 1)  # (512->1) 16->16

        return out


