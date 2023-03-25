#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from utils import CharsetMapper
from .backbone.resnet45 import resnet45
from .backbone.swin_transformer import SwinTransformer
from .decoder.attentional_parallel_decoder import PositionAttention
import torch.nn.functional as F


class Projection(nn.Module):
    """
    Creates projection head
    Args:
    n_in (int): Number of input features
    n_hidden (int): Number of hidden features
    n_out (int): Number of output features
    use_bn (bool): Whether to use batch norm
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 use_bn: bool = True):
        super().__init__()

        # No point in using bias if we've batch norm
        self.lin1 = nn.Linear(n_in, n_hidden, bias=not use_bn)
        self.bn = nn.LayerNorm(n_hidden) if use_bn else nn.Identity()
        self.relu = nn.ReLU()
        # No bias for the final linear layer
        self.lin2 = nn.Linear(n_hidden, n_out, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input: visual feature [N, T, E]
        output: contextual feature [N, T, output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)      # N, T, E --> N, T, 2 * hidden_size
        output = self.linear(recurrent)     # N, T, output_size
        return output


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.charset = CharsetMapper(config.dataset_charset_path, max_length=config.dataset_max_length + 1)

        self.imgH, self.imgW = config.dataset_imgH, config.dataset_imgW
        self.loss_weight = config.optimizer_loss_weight
        self.d_model = config.model_d_model

        self.backbone_type = config.model_backbone

        if config.model_backbone == 'resnet45':
            self.backbone = resnet45()
        elif config.model_backbone == 'swin':
            self.SwinT_patch_size = (1, 4)
            self.SwinT_window_size = (2, 8)
            self.SwinT_embedding_dim = 96
            self.SwinT_depths = (2, 2, 18)
            self.SwinT_num_heads = (3, 6, 12)

            self.backbone = SwinTransformer(img_size=(self.imgH, self.imgW), patch_size=self.SwinT_patch_size,
                                            window_size=self.SwinT_window_size, embed_dim=self.SwinT_embedding_dim,
                                            depths=self.SwinT_depths, num_heads=self.SwinT_num_heads)

            self.conv = nn.Conv2d(in_channels=self.SwinT_embedding_dim * 4, out_channels=self.d_model,
                                  kernel_size=(3, 3),
                                  stride=(1, 1), padding=(1, 1))

        self.decoder = PositionAttention(max_length=config.dataset_max_length + 1,
                                         in_channels=self.d_model, num_channels=self.d_model // 4,
                                         h=self.imgH // 8, w=self.imgW // 8)

        self.cls = nn.Linear(in_features=self.d_model, out_features=self.charset.num_classes)

        self.apply(self._init_weights)

        self.contrastive = config.global_contrastive
        self.projection_type = config.global_projection_type

        if self.contrastive:
            if self.projection_type == 'projection_free':
                pass
            elif self.projection_type == 'wordArt':
                # WordArt projection head, 2-MLP
                self.head = nn.Sequential(
                    nn.Linear(512, 2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048, 512)
                )
            elif self.projection_type == 'SimCLR':
                # SimCLR
                self.head = Projection(512, 2048, 512, True)
            elif self.projection_type == 'BiLSTM':
                # SeqCLR, Bi-LSTM as projection head
                self.head = BidirectionalLSTM(input_size=self.d_model,
                                              hidden_size=self.d_model,
                                              output_size=self.d_model)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_length(self, logit, dim=-1):
        """ Greed decoder to obtain length from logit"""
        out = (logit.argmax(dim=-1) == self.charset.null_label)
        abn = out.any(dim)
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1  # additional end token
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        return out

    def _single_image_forward(self, image, is_training=True):
        features = {}
        feature = self.backbone(image)
        if self.backbone_type == 'resnet45':
            shortcut = feature
            B, C, H, W = feature.shape
            features['backbone_feature'] = feature
            feature = shortcut.view(B, self.d_model, -1).permute(0, 2, 1)

        elif self.backbone_type == 'swin':
            feature = feature.permute(0, 2, 1)
            feature = feature.view(feature.size(0), feature.size(1), self.imgH // 8, self.imgW // 8).contiguous()
            features['backbone_feature'] = feature
            feature = self.conv(feature).view(feature.size(0), self.d_model, -1).permute(0, 2, 1)

        attn_vecs, attn_scores = self.decoder(
            feature.permute(0, 2, 1).view(feature.size(0), self.d_model, self.imgH // 8, self.imgW // 8))

        con_feature = None
        if self.contrastive and is_training:
            if self.projection_type == 'projection_free':
                # projection-free
                con_feature = attn_vecs
            elif self.projection_type == 'wordArt' or self.projection_type == 'SimCLR':
                # add projection
                N, T, E = attn_vecs.shape
                con_feature = attn_vecs.reshape(-1, E)
                con_feature = self.head(con_feature)
                con_feature = F.normalize(con_feature, dim=1)
                con_feature = con_feature.reshape(N, T, E)
            elif self.projection_type == 'BiLSTM':
                con_feature = self.head(attn_vecs)

        logits = self.cls(attn_vecs)
        pt_lengths = self._get_length(logits)

        return attn_vecs, attn_scores, logits, pt_lengths, con_feature

    def forward(self, images, *args):
        if not self.contrastive:
            attn_vecs, attn_scores, logits, pt_lengths, _ = self._single_image_forward(images)
            return {'feature': attn_vecs,
                    'logits': logits,
                    'pt_lengths': pt_lengths,
                    'attn_score': attn_scores,
                    'loss_weight': self.loss_weight,
                    }
        elif self.contrastive:
            if len(images.shape) > 4:
                image_1, image_2 = images[:, 0], images[:, 1]
                bsz = image_1.shape[0]

                attn_vecs, attn_scores, logits, pt_lengths, con_features = self._single_image_forward(
                    torch.cat([image_1, image_2], dim=0))

                attn_vecs_1, attn_vecs_2 = torch.split(attn_vecs, [bsz, bsz], dim=0)
                attn_scores_1, attn_scores_2 = torch.split(attn_scores, [bsz, bsz], dim=0)
                logits_1, logits_2 = torch.split(logits, [bsz, bsz], dim=0)
                pt_lengths_1, pt_lengths_2 = torch.split(pt_lengths, [bsz, bsz], dim=0)

                con_feature_1, con_feature_2 = torch.split(con_features, [bsz, bsz], dim=0)

                outputs_view_1 = {
                    'feature': attn_vecs_1,
                    'con_feature_view_1': con_feature_1,
                    'logits': logits_1,
                    'pt_lengths': pt_lengths_1,
                    'attn_score': attn_scores_1,
                }

                outputs_view_2 = {
                    'feature': attn_vecs_2,
                    'con_feature_view_2': con_feature_2,
                    'logits': logits_2,
                    'pt_lengths': pt_lengths_2,
                    'attn_score': attn_scores_2,
                }

                return {'outputs_view_1': outputs_view_1,
                        'outputs_view_2': outputs_view_2,
                        'loss_weight': self.loss_weight,
                        }

            else:
                attn_vecs, attn_scores, logits, pt_lengths, _ = self._single_image_forward(images, is_training=False)

                return {'feature': attn_vecs,
                        'logits': logits,
                        'pt_lengths': pt_lengths,
                        'attn_score': attn_scores,
                        'loss_weight': self.loss_weight,
                        }

