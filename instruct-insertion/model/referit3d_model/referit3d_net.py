import math
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from openpoints.models.backbone.pointnext import PointNextEncoder
from torch import nn
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
)

from .utils import get_siamese_features, my_get_siamese_features


def optional_repeat(value, times):
    """helper function, to repeat a parameter's value many times
    :param value: an single basic python type (int, float, boolean, string), or a list with length equals to times
    :param times: int, how many times to repeat
    :return: a list with length equal to times
    """
    if type(value) is not list:
        value = [value]

    if len(value) != 1 and len(value) != times:
        raise ValueError("The value should be a singleton, or be a list with times length.")

    if len(value) == times:
        return value  # do nothing

    return np.array(value).repeat(times).tolist()


class MLP(nn.Module):
    """Multi-linear perceptron. That is a k-layer deep network where each layer is a fully-connected layer, with
    (optionally) batch-norm, a non-linearity and dropout. The last layer (output) is always a 'pure' linear function.
    """

    def __init__(
        self,
        in_feat_dims,
        out_channels,
        dropout_rate=0,
        non_linearity=nn.ReLU(inplace=True),
        closure=None,
        norm_type=nn.BatchNorm1d,
    ):
        """Constructor
        :param in_feat_dims: input feature dimensions
        :param out_channels: list of ints describing each the number hidden/final neurons. The
        :param b_norm: True/False, or list of booleans
        :param dropout_rate: int, or list of int values
        :param non_linearity: nn.Module
        :param closure: optional nn.Module to use at the end of the MLP
        """
        super(MLP, self).__init__()

        n_layers = len(out_channels)
        dropout_rate = optional_repeat(dropout_rate, n_layers - 1)

        previous_feat_dim = in_feat_dims
        all_ops = []

        for depth in range(len(out_channels)):
            out_dim = out_channels[depth]
            affine_op = nn.Linear(previous_feat_dim, out_dim, bias=True)
            all_ops.append(affine_op)

            if depth < len(out_channels) - 1:
                if norm_type is not None:
                    all_ops.append(norm_type(out_dim))

                if non_linearity is not None:
                    all_ops.append(non_linearity)

                if dropout_rate[depth] > 0:
                    all_ops.append(nn.Dropout(p=dropout_rate[depth]))

            previous_feat_dim = out_dim

        if closure is not None:
            all_ops.append(closure)

        self.net = nn.Sequential(*all_ops)

    def forward(self, x):
        return self.net(x)


class ReferIt3DNet_transformer(nn.Module):
    def __init__(self, args, n_obj_classes, class_name_tokens, ignore_index):
        super().__init__()

        self.bert_pretrain_path = args.bert_pretrain_path

        self.view_number = args.view_number
        self.rotate_number = args.rotate_number

        self.label_lang_sup = args.label_lang_sup
        self.aggregate_type = args.aggregate_type

        self.encoder_layer_num = args.encoder_layer_num
        self.decoder_layer_num = args.decoder_layer_num
        self.decoder_nhead_num = args.decoder_nhead_num

        self.object_dim = args.object_latent_dim
        self.inner_dim = args.inner_dim

        self.dropout_rate = args.dropout_rate
        self.lang_cls_alpha = args.lang_cls_alpha
        self.obj_cls_alpha = args.obj_cls_alpha

        # ADD Point BERT
        self.obj_encoder = PointNextEncoder(
            in_channels=7,
            width=32,
            blocks=[1, 3, 5, 3, 3],
            strides=[1, 4, 4, 4, 4],
            nsample=32,
            radius=0.05,
            aggr_args=dict(feature_type="dp_fj", reduction="max"),
            group_args=dict(NAME="ballquery", normalize_dp=True),
            sa_layers=1,
            sa_use_res=False,
            expansion=4,
            conv_args=dict(order="onv-norm-act"),
            act_args=dict(act="relu"),
            norm_args=dict(norm="bn"),
        )
        self.point_trans = False

        self.language_encoder = BertModel.from_pretrained(self.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[
            : self.encoder_layer_num
        ]

        self.refer_encoder = nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=self.inner_dim,
                nhead=self.decoder_nhead_num,
                dim_feedforward=2048,
                activation="gelu",
            ),
            num_layers=self.decoder_layer_num,
        )

        # Classifier heads
        self.language_clf = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.inner_dim, n_obj_classes),
        )

        self.object_language_clf = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.inner_dim, 1),
        )

        if not self.label_lang_sup:
            self.obj_clf = MLP(
                self.inner_dim,
                [self.object_dim, self.object_dim, n_obj_classes],
                dropout_rate=self.dropout_rate,
            )

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(4, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.class_name_tokens = class_name_tokens

        self.box_layers = MLP(
            self.inner_dim,
            [self.inner_dim, self.inner_dim, 4],
            dropout_rate=self.dropout_rate,
        )

        self.locate_loss = nn.L1Loss()
        self.dist_loss = nn.L1Loss()
        # self.dist_loss = nn.MSELoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
        input_points = input_points.float().to(self.device)
        box_infos = box_infos.float().to(self.device)
        xyz = input_points[:, :, :, :3]
        bxyz = box_infos[:, :, :3]  # B,N,3
        B, N, P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor(
            [i * 2.0 * np.pi / self.rotate_number for i in range(self.rotate_number)]
        ).to(self.device)
        view_theta_arr = torch.Tensor(
            [i * 2.0 * np.pi / self.view_number for i in range(self.view_number)]
        ).to(self.device)

        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            theta = rotate_theta_arr[
                torch.randint(0, self.rotate_number, (B,))
            ]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = (
                torch.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
                .to(self.device)[None]
                .repeat(B, 1, 1)
            )
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[:, :, :, :3] = torch.matmul(
                xyz.reshape(B, N * P, 3), rotate_matrix
            ).reshape(B, N, P, 3)
            bxyz = torch.matmul(bxyz.reshape(B, N, 3), rotate_matrix).reshape(B, N, 3)

        # multi-view
        bsize = box_infos[:, :, -1:]
        boxs = []
        for theta in view_theta_arr:
            rotate_matrix = torch.Tensor(
                [
                    [math.cos(theta), -math.sin(theta), 0.0],
                    [math.sin(theta), math.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B * N, 3), rotate_matrix).reshape(B, N, 3)
            boxs.append(torch.cat([rxyz, bsize], dim=-1))
        boxs = torch.stack(boxs, dim=1)
        return input_points, boxs

    def compute_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOCATE_PREDS, AUX_LOGITS=None):
        # CLASS_LOGITS.transpose(2, 1) (B,C=525,D=52) <--> batch['class_labels'] (B,D)
        obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch["ctx_class"])

        # LANG_LOGITS (B, C=524) <--> batch['target_class'] (B,)
        lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch["tgt_class"])

        # center loss
        # LOCATE_PREDS[:, :3] (B,3) <--> batch['tgt_box_center'] (B,3)
        locate_loss = self.locate_loss(LOCATE_PREDS[:, :3], batch["tgt_box_center"])

        # dist loss
        # LOCATE_PREDS[:, -1] (B,) <--> batch['tgt_box_center'] (B,)
        dist_loss = self.dist_loss(LOCATE_PREDS[:, -1], batch["tgt_box_max_dist"])

        total_loss = (
            locate_loss
            + dist_loss
            + self.obj_cls_alpha * obj_clf_loss
            + self.lang_cls_alpha * lang_clf_loss
        )

        return total_loss

    def first_stage_forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ## rotation augmentation and multi_view generation
        obj_points, boxs = self.aug_input(
            batch["ctx_pc"],
            torch.concat(batch["ctx_box_center"], batch["ctx_box_max_dist"][:, :, None], dim=-1),
        )
        B, N, P = obj_points.shape[:3]

        ## obj_encoding
        objects_features = get_siamese_features(
            self.obj_encoder,
            obj_points,
            aggregator=torch.stack,
            batch_pnet=True,
        )

        ## obj_encoding
        obj_feats = self.obj_feature_mapping(objects_features)
        box_infos = self.box_feature_mapping(boxs)
        obj_infos = obj_feats[:, None].repeat(1, self.view_number, 1, 1) + box_infos

        # <LOSS>: obj_cls
        if self.label_lang_sup:
            label_lang_infos = self.language_encoder(**self.class_name_tokens)[0][:, 0]
            CLASS_LOGITS = torch.matmul(
                obj_feats.reshape(B * N, -1), label_lang_infos.permute(1, 0)
            ).reshape(B, N, -1)
        else:
            CLASS_LOGITS = self.obj_clf(obj_feats.reshape(B * N, -1)).reshape(B, N, -1)

        ## language_encoding
        lang_tokens = batch["tokens"]
        lang_infos = self.language_encoder(**lang_tokens)[0]

        # <LOSS>: lang_cls
        lang_features = lang_infos[:, 0]
        LANG_LOGITS = self.language_clf(lang_features)

        ## multi-modal_fusion
        cat_infos = obj_infos.reshape(B * self.view_number, -1, self.inner_dim)
        mem_infos = (
            lang_infos[:, None]
            .repeat(1, self.view_number, 1, 1)
            .reshape(B * self.view_number, -1, self.inner_dim)
        )
        out_feats = (
            self.refer_encoder(cat_infos.transpose(0, 1), mem_infos.transpose(0, 1))
            .transpose(0, 1)
            .reshape(B, self.view_number, -1, self.inner_dim)
        )

        # Returns: (B, V, N, C), (B, N, # of classes), (B, C)
        return out_feats, CLASS_LOGITS, LANG_LOGITS

    def second_stage_forward(
        self, out_feats, batch, CLASS_LOGITS, LANG_LOGITS
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, V, N = out_feats.shape[:3]
        ## view_aggregation
        refer_feat = out_feats
        if self.aggregate_type == "avg":
            agg_feats = (refer_feat / self.view_number).sum(dim=1)
        elif self.aggregate_type == "avgmax":
            agg_feats = (refer_feat / self.view_number).sum(dim=1) + refer_feat.max(dim=1).values
        else:
            agg_feats = refer_feat.max(dim=1).values

        # return the center point of box and box max distance
        # FIXME: the shape is (B, N, 4), but we dont want the `N` here. Check other detection tasks for ideas.
        LOCATE_PREDS = self.box_layers(agg_feats.reshape(B * N, -1)).reshape(B, N, -1)
        LOSS = self.compute_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOCATE_PREDS)

        return LOSS, LOCATE_PREDS
