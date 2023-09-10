import math
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from openpoints.models.backbone.pointnext import PointNextEncoder
from torch import nn
from transformers import (
    BatchEncoding,
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
    def __init__(self, args, n_obj_classes, class_name_tokens: BatchEncoding, ignore_index: int):
        super().__init__()

        self.bert_pretrain_path = args.bert_pretrain_path

        self.view_number = args.view_number
        self.rotate_number = args.rotate_number

        self.label_lang_sup = args.label_lang_sup

        self.encoder_layer_num = args.encoder_layer_num
        self.decoder_layer_num = args.decoder_layer_num
        self.decoder_nhead_num = args.decoder_nhead_num

        self.inner_dim = args.inner_dim

        self.dropout_rate = args.dropout_rate
        self.lang_cls_alpha = args.lang_cls_alpha
        self.obj_cls_alpha = args.obj_cls_alpha
        self.rel_pred_alpha = args.rel_pred_alpha

        self.class_name_tokens = class_name_tokens

        self.height_append: bool = args.height_append

        self.obj_encoder = PointNextEncoder(
            in_channels=7,
            width=32,
            blocks=[1, 3, 5, 3, 3],
            strides=[1, 4, 4, 4, 4],
            nsample=32,
            radius=0.05,
            aggr_args=edict({"feature_type": "dp_fj", "reduction": "max"}),
            group_args=edict({"NAME": "ballquery", "normalize_dp": True}),
            sa_layers=1,
            sa_use_res=False,
            expansion=4,
            conv_args=edict({"order": "conv-norm-act"}),
            act_args=edict({"act": "relu", "inplace": True}),
            norm_args=edict({"norm": "bn"}),
        )
        self.obj_encoder_agg_proj = nn.Linear(
            4 * (args.points_per_object // 2)
            if self.height_append
            else 3 * (args.points_per_object // 2),
            self.inner_dim,
        )

        self.language_encoder = BertModel.from_pretrained(self.bert_pretrain_path)
        # leave only part of the encoder layers
        self.language_encoder.encoder.layer = self.language_encoder.encoder.layer[
            : self.encoder_layer_num
        ]
        # freeze the embedding layer and first layer
        self.language_encoder.embeddings.requires_grad_(False)
        self.language_encoder.encoder.layer[0].requires_grad_(False)

        self.offset: bool = args.offset_prediction
        if self.offset:
            self.offset_layer = MLP(
                self.inner_dim,
                [self.inner_dim, self.inner_dim, 3],
                dropout_rate=self.dropout_rate,
                norm_type=None,
            )

        self.refer_encoder = nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=self.inner_dim,
                nhead=self.decoder_nhead_num,
                dim_feedforward=2048,
                activation="gelu",
                batch_first=True,
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
                [self.inner_dim, self.inner_dim, n_obj_classes],
                dropout_rate=self.dropout_rate,
            )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(4, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_layers = MLP(
            self.inner_dim,
            [self.inner_dim, self.inner_dim, 4],
            dropout_rate=self.dropout_rate,
            norm_type=None,
        )

        self.locate_token = nn.Embedding(1, self.inner_dim)

        self.locate_loss = nn.MSELoss()
        self.radius_loss = nn.L1Loss()

        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
        import warnings

        warnings.warn("`aug_input` is deprecated.", DeprecationWarning)
        input_points = input_points.float()
        box_infos = box_infos.float()
        device = input_points.device
        xyz = input_points[:, :, :, :3]
        bxyz = box_infos[:, :, :3]  # B,N,3
        B, N, P = xyz.shape[:3]
        V = self.view_number
        rotate_theta_arr = torch.tensor(
            [i * 2.0 * np.pi / self.rotate_number for i in range(self.rotate_number)],
            device=device,
        )
        view_theta_arr = torch.tensor(
            [i * 2.0 * np.pi / V for i in range(V)],
            device=device,
        )

        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            theta = rotate_theta_arr[
                torch.randint(0, self.rotate_number, (B,))
            ]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device
            )[None].repeat(B, 1, 1)
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
            rotate_matrix = torch.tensor(
                [
                    [math.cos(theta), -math.sin(theta), 0.0],
                    [math.sin(theta), math.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
            )
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
        if self.offset:
            locate_loss = (1.0 - self.rel_pred_alpha) * self.locate_loss(
                LOCATE_PREDS[:, :3], batch["tgt_box_center"]
            )
        else:
            locate_loss = self.locate_loss(LOCATE_PREDS[:, :3], batch["tgt_box_center"])

        # radius loss
        # LOCATE_PREDS[:, -1] (B,) <--> batch['tgt_box_max_dist'] (B,)
        dist_loss = self.radius_loss(LOCATE_PREDS[:, -1], batch["tgt_box_max_dist"])

        total_loss = (
            locate_loss
            + dist_loss
            + self.obj_cls_alpha * obj_clf_loss
            + self.lang_cls_alpha * lang_clf_loss
        )

        return total_loss

    def forward(self, batch: dict) -> Tuple[torch.Tensor, ...]:
        ###################################
        #                                 #
        #    points/views augmentation    #
        #                                 #
        ###################################
        # NOTE: stop multi view temporarily
        # obj_points, boxs = self.aug_input(
        #     batch["ctx_pc"],
        #     torch.concat((batch["ctx_box_center"], batch["ctx_box_max_dist"][:, :, None]), dim=-1),
        # )
        obj_points = batch["ctx_pc"].float()
        boxs = torch.cat(
            (batch["ctx_box_center"], batch["ctx_box_max_dist"][:, :, None]), dim=-1
        ).float()  # (B, N, 4)
        boxs = boxs[:, None].repeat(1, self.view_number, 1, 1)  # (B, V, N, 4)
        B, N, P, D = obj_points.shape
        V = self.view_number

        ######################
        #                    #
        #    obj encoding    #
        #                    #
        ######################
        obj_feats, CLASS_LOGITS = self.forward_obj_cls(obj_points)

        box_infos = self.box_feature_mapping(boxs)
        obj_infos = obj_feats[:, None].repeat(1, V, 1, 1) + box_infos

        ###########################
        #                         #
        #    language encoding    #
        #                         #
        ###########################
        lang_tokens = batch["tokens"]
        lang_infos: torch.Tensor = self.language_encoder(**lang_tokens)[0]  # (B, # of tokens, C)

        # <LOSS>: lang_cls
        LANG_LOGITS = self.language_clf(lang_infos[:, 0])

        ############################
        #                          #
        #    multi-modal fusion    #
        #                          #
        ############################
        # mask generation
        lang_mask: torch.BoolTensor = batch["tokens"]["attention_mask"] == 0  # (B, # of tokens)
        lang_mask = lang_mask[:, None].repeat(1, V, 1).reshape(B * V, -1)  # (B * V, # of tokens)
        obj_mask: torch.BoolTensor = batch["ctx_key_padding_mask"]  # (B, N)
        # append first token mask
        obj_mask_w_tgt = torch.cat(
            [torch.zeros((B, 1), dtype=torch.bool, device=obj_mask.device), obj_mask], dim=1
        )  # (B, N+1)
        obj_mask_w_tgt = obj_mask_w_tgt[:, None].repeat(1, V, 1).reshape(B * V, -1)  # (B * V, N+1)

        # feature prepare
        cat_infos = obj_infos.reshape(B * V, -1, self.inner_dim)
        cat_infos = torch.cat(
            [self.locate_token.weight[None].repeat(B * V, 1, 1), cat_infos], dim=1
        )  # (B * V, N+1, C)

        mem_infos = (
            lang_infos[:, None].repeat(1, V, 1, 1).reshape(B * V, -1, self.inner_dim)
        )  # (B * V, # of tokens, C)
        out_feats = self.refer_encoder(
            tgt=cat_infos,
            memory=mem_infos,
            tgt_key_padding_mask=obj_mask_w_tgt,
            memory_key_padding_mask=lang_mask,
        ).reshape(
            B, V, -1, self.inner_dim
        )  # (B, V, N+1, C)

        out_feats = out_feats.mean(dim=1)  # (B, N+1, C)
        ctx_embeds = out_feats[:, 0, :]  # (B, V, C)

        ######################
        #                    #
        #    loss compute    #
        #                    #
        ######################
        # return the center point of box and box max distance
        # ctx_embeds: (B, C)

        # directly predict the center point of box
        LOCATE_PREDS = self.box_layers(ctx_embeds)  # (B, 4)

        LOSS = self.compute_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOCATE_PREDS)

        if self.offset:
            # predict the offset
            obj_mask_coef = (~obj_mask).float()  # (B, N)
            coords = batch["ctx_box_center"]  # (B, N, 3)
            offset_coords = self.offset_layer(out_feats[:, 1:, :])  # (B, N, 3)
            pos_preds = coords + offset_coords
            POS_PRED_LOSS = F.mse_loss(
                pos_preds, batch["tgt_box_center"][:, None].repeat(1, N, 1), reduction="none"
            )  # (B, N, 3)
            POS_PRED_LOSS = POS_PRED_LOSS.mean(dim=-1) * obj_mask_coef  # (B, N)
            POS_PRED_LOSS = POS_PRED_LOSS.sum(dim=-1) / obj_mask_coef.sum(dim=-1)  # (B,)
            LOSS += self.rel_pred_alpha * POS_PRED_LOSS.mean()
            # adjust the box center
            mean_pos_preds = pos_preds * obj_mask_coef[:, :, None]  # (B, N, 3)
            mean_pos_preds = mean_pos_preds.sum(dim=1) / obj_mask_coef.sum(dim=1)[:, None]  # (B, 3)
            LOCATE_PREDS[:, :3] = (1.0 - self.rel_pred_alpha) * LOCATE_PREDS[:, :3] + (
                self.rel_pred_alpha
            ) * mean_pos_preds

        # Returns: (B, C), (,), (B, N, # of classes), (B, C), (B, 4)
        return ctx_embeds, LOSS, CLASS_LOGITS.detach(), LANG_LOGITS.detach(), LOCATE_PREDS.detach()

    def forward_obj_cls(self, obj_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obj_points (torch.Tensor): (B, N, P, D)

        Returns:
            (torch.Tensor): (B, N, C)
            (torch.Tensor): (B, N, # of classes)
        """
        B, N, P, D = obj_points.shape
        obj_feats = self.obj_encoder.forward_cls_feat(
            {
                "pos": obj_points[:, :, :, :3].reshape(B * N, P, -1).contiguous(),
                "x": obj_points[:, :, :, :].reshape(B * N, P, -1).transpose(1, 2).contiguous(),
            }
        )  # (B * N, 512, 4)
        obj_feats = self.obj_encoder_agg_proj(obj_feats.reshape(B * N, -1)).reshape(
            B, N, -1
        )  # (B, N, C)

        # <LOSS>: obj_cls
        if self.label_lang_sup:
            label_lang_infos = self.language_encoder(**self.class_name_tokens)[0][:, 0]
            CLASS_LOGITS = torch.matmul(
                obj_feats.reshape(B * N, -1), label_lang_infos.permute(1, 0)
            ).reshape(B, N, -1)
        else:
            CLASS_LOGITS = self.obj_clf(obj_feats.reshape(B * N, -1)).reshape(B, N, -1)

        return obj_feats, CLASS_LOGITS
