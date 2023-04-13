# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from ..builder import ROTATED_HEADS, build_head
from .rotated_anchor_head import RotatedAnchorHead


@ROTATED_HEADS.register_module()
class EoodHead(RotatedAnchorHead):
    r"""An anchor-based head used in `RotatedRetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 predictors = [],
                 parallel = True,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     ),
                 **kwargs):
        self.predictors = [build_head(p) for p in predictors]
        assert len(self.predictors) > 0, "need a predictor"  
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.parallel = parallel
        super(EoodHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.predictors = nn.ModuleList(self.predictors)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

    def init_weights(self):
        super().init_weights()
        for p in self.predictors:
            # assert isinstance(p, BaseHeadPredictor)
            if hasattr(p, 'init_weights'):
                p.init_weights()

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        
        return cls_feat, reg_feat

    def forward(self, feats):
        # feats : from fpn 
        multi_level_features = multi_apply(self.forward_single, feats)
        multi_level_features = [(multi_level_features[0][i], multi_level_features[1][i]) for i in range(len(feats)) ]
        if not self.training:
            cls_score, bbox_pred = multi_apply(self.predictors[0].forward_single, multi_level_features)
            return cls_score, bbox_pred
        else:
            preds = [p.forward(multi_level_features) for p in self.predictors]
            return (tuple(preds),)
            
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        assert len(preds) == len(self.predictors), f'The number of predictions is not equal to the number of predictors.'
        loss_dict = dict()
        if self.parallel:
            for i, (pred, predictor) in enumerate(zip(preds, self.predictors)):
                num_loss = len(loss_dict)
                # cls_scores, bbox_preds = pred
                loss_i = predictor.loss(
                                        *pred,
                                        gt_bboxes,
                                        gt_labels,
                                        img_metas,
                                        gt_bboxes_ignore=gt_bboxes_ignore)
                for key, val in loss_i.items():
                    loss_dict[f'loss_{i}_{key}'] = val
                assert (num_loss + len(loss_i)) == len(loss_dict), f'Some losses are overwritten.'
            return loss_dict
        else:  # cascade
            assert len(self.predictors) > 1, "if cascade, aux head need more than one"
            main_predictor = self.predictors[0]
            main_pred = preds[0]
            loss_main = main_predictor.loss(
                                        *main_pred,
                                        gt_bboxes,
                                        gt_labels,
                                        img_metas,
                                        gt_bboxes_ignore=gt_bboxes_ignore)
            for name, value in loss_main.items():
                loss_dict[f'loss_main_.{name}'] = value

            aux_base_predictor = self.predictors[1]
            aux_base_pred = preds[1]
            loss_aux_base = aux_base_predictor.loss(
                                        *aux_base_pred,
                                        gt_bboxes,
                                        gt_labels,
                                        img_metas,
                                        gt_bboxes_ignore=gt_bboxes_ignore)
            for name, value in loss_aux_base.items():
                loss_dict[f'loss_aux_base_.{name}'] = value

            rois = aux_base_predictor.refine_bboxes(*aux_base_pred)
            for i, (pred, predictor) in enumerate(zip(preds[2:], self.predictors[2:])):
                loss_i = predictor.loss(
                                        *pred,
                                        gt_bboxes,
                                        gt_labels,
                                        img_metas,
                                        gt_bboxes_ignore=gt_bboxes_ignore,
                                        rois=rois)
                for key, val in loss_i.items():
                    loss_dict[f'loss_aux_{i}_{key}'] = val

                if i + 1 in range(len(self.predictors)-2):
                    rois = predictor.refine_bboxes(*pred, rois=rois)
            return loss_dict


    # first head only
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        return self.predictors[0].get_bboxes(cls_scores,
                                            bbox_preds,
                                            img_metas,
                                            cfg=cfg,
                                            rescale=rescale,
                                            with_nms=with_nms)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_bboxes(self, cls_scores, bbox_preds):
        """Filter predicted bounding boxes at each position of the feature
        maps. Only one bounding boxes with highest score will be left at each
        position. This filter will be used in R3Det prior to the first feature
        refinement stage.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level \
                of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors,
                                          self.cls_out_channels)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)
            best_ind = best_ind.expand(-1, -1, -1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)
            best_pred = bbox_pred.gather(
                dim=-2, index=best_ind).squeeze(dim=-2)

            anchors = anchors.reshape(-1, self.num_anchors, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]
                best_pred_i = best_pred[img_id]
                best_anchor_i = anchors.gather(
                    dim=-2, index=best_ind_i).squeeze(dim=-2)
                best_bbox_i = self.bbox_coder.decode(best_anchor_i,
                                                     best_pred_i)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds):
        """This function will be used in S2ANet, whose num_anchors=1.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)

        Returns:
            list[list[Tensor]]: refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 5)
            anchors = mlvl_anchors[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(anchors, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list
