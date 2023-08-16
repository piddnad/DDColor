import torch
import torch.nn as nn

from basicsr.archs.ddcolor_arch_utils.unet import Hook, CustomPixelShuffle_ICNR,  UnetBlockWide, NormType, custom_conv_layer
from basicsr.archs.ddcolor_arch_utils.convnext import ConvNeXt
from basicsr.archs.ddcolor_arch_utils.transformer_utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from basicsr.archs.ddcolor_arch_utils.position_encoding import PositionEmbeddingSine
from basicsr.archs.ddcolor_arch_utils.transformer import Transformer
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DDColor(nn.Module):

    def __init__(self,
                 encoder_name='convnext-l',
                 decoder_name='MultiScaleColorDecoder',
                 num_input_channels=3,
                 input_size=(256, 256),
                 nf=512,
                 num_output_channels=3,
                 last_norm='Weight',
                 do_normalize=False,
                 num_queries=256,
                 num_scales=3,
                 dec_layers=9,
                 encoder_from_pretrain=False):
        super().__init__()

        self.encoder = Encoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'], from_pretrain=encoder_from_pretrain)
        self.encoder.eval()
        test_input = torch.randn(1, num_input_channels, *input_size)
        self.encoder(test_input)

        self.decoder = Decoder(
            self.encoder.hooks,
            nf=nf,
            last_norm=last_norm,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
            decoder_name=decoder_name
        )
        self.refine_net = nn.Sequential(custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False, norm_type=NormType.Spectral))
    
        self.do_normalize = do_normalize
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.normalize(x)
        
        self.encoder(x)
        out_feat = self.decoder()
        coarse_input = torch.cat([out_feat, x], dim=1)
        out = self.refine_net(coarse_input)

        if self.do_normalize:
            out = self.denormalize(out)
        return out


class Decoder(nn.Module):

    def __init__(self,
                 hooks,
                 nf=512,
                 blur=True,
                 last_norm='Weight',
                 num_queries=256,
                 num_scales=3,
                 dec_layers=9,
                 decoder_name='MultiScaleColorDecoder'):
        super().__init__()
        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)
        self.decoder_name = decoder_name

        self.layers = self.make_layers()
        embed_dim = nf // 2

        self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)
        
        if self.decoder_name == 'MultiScaleColorDecoder':
            self.color_decoder = MultiScaleColorDecoder(
                in_channels=[512, 512, 256],
                num_queries=num_queries,
                num_scales=num_scales,
                dec_layers=dec_layers,
            )
        else:
            self.color_decoder = SingleColorDecoder(
                in_channels=hooks[-1].feature.shape[1], 
                num_queries=num_queries,
            )


    def forward(self):
        encode_feat = self.hooks[-1].feature
        out0 = self.layers[0](encode_feat)
        out1 = self.layers[1](out0) 
        out2 = self.layers[2](out1) 
        out3 = self.last_shuf(out2) 

        if self.decoder_name == 'MultiScaleColorDecoder':
            out = self.color_decoder([out0, out1, out2], out3)
        else:
            out = self.color_decoder(out3, encode_feat)
           
        return out

    def make_layers(self):
        decoder_layers = []

        e_in_c = self.hooks[-1].feature.shape[1]
        in_c = e_in_c

        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[1]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlockWide(
                    in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral))
            in_c = out_c
        return nn.Sequential(*decoder_layers)


class Encoder(nn.Module):

    def __init__(self, encoder_name, hook_names, from_pretrain, **kwargs):
        super().__init__()
 
        if encoder_name == 'convnext-t' or encoder_name == 'convnext':
            self.arch = ConvNeXt()
        elif encoder_name == 'convnext-s':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
        elif encoder_name == 'convnext-b':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        elif encoder_name == 'convnext-l':
            self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        else:
            raise NotImplementedError

        self.encoder_name = encoder_name
        self.hook_names = hook_names
        self.hooks = self.setup_hooks()

        if from_pretrain:
            self.load_pretrain_model()

    def setup_hooks(self):
        hooks = [Hook(self.arch._modules[name]) for name in self.hook_names]
        return hooks

    def forward(self, x):
        return self.arch(x)
    
    def load_pretrain_model(self):
        if self.encoder_name == 'convnext-t' or self.encoder_name == 'convnext':
            self.load('pretrain/convnext_tiny_22k_224.pth')
        elif self.encoder_name == 'convnext-s':
            self.load('pretrain/convnext_small_22k_224.pth')
        elif self.encoder_name == 'convnext-b':
            self.load('pretrain/convnext_base_22k_224.pth')
        elif self.encoder_name == 'convnext-l':
            self.load('pretrain/convnext_large_22k_224.pth')
        else:
            raise NotImplementedError
        print('Loaded pretrained convnext model.')

    def load(self, path):
        from basicsr.utils import get_root_logger
        logger = get_root_logger()
        if not path:
            logger.info("No checkpoint found. Initializing model from scratch")
            return
        logger.info("[Encoder] Loading from {} ...".format(path))
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        checkpoint_state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
        incompatible = self.arch.load_state_dict(checkpoint_state_dict, strict=False)

        if incompatible.missing_keys:
            msg = "Some model parameters or buffers are not found in the checkpoint:\n"
            msg += str(incompatible.missing_keys)
            logger.warning(msg)
        if incompatible.unexpected_keys:
            msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
            msg += str(incompatible.unexpected_keys)
            logger.warning(msg)


class MultiScaleColorDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=9,
        pre_norm=False,
        color_embed_dim=256,
        enforce_input_project=True,
        num_scales=3
    ):
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable color query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable color query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = num_scales
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # input projections
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []

        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)    

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [N, bs, C]  -> [bs, N, C]
        color_embed = self.color_embed(decoder_output)
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

        return out


class SingleColorDecoder(nn.Module):

    def __init__(
        self,
        in_channels=768, 
        hidden_dim=256,
        num_queries=256,  # 100
        nheads=8,
        dropout=0.1,
        dim_feedforward=2048,
        enc_layers=0,
        dec_layers=6,
        pre_norm=False,
        deep_supervision=True,
        enforce_input_project=True,
    ):
        
        super().__init__()

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
        self.num_queries = num_queries
        self.transformer = transformer
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            nn.init.kaiming_uniform_(self.input_proj.weight, a=1)
            if self.input_proj.bias is not None:
                nn.init.constant_(self.input_proj.bias, 0)
        else:
            self.input_proj = nn.Sequential()


    def forward(self, img_features, encode_feat):
        pos = self.pe_layer(encode_feat)
        src = encode_feat
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)
        color_embed = hs[-1]
        color_preds = torch.einsum('bqc,bchw->bqhw', color_embed, img_features)
        return color_preds

