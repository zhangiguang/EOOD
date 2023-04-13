# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from ..iou_calculators import rbbox_overlaps
from ..builder import ROTATED_BBOX_ASSIGNERS

from ..transforms import obb2hbb, obb2xyxy


@ROTATED_BBOX_ASSIGNERS.register_module()            
class PolaAssigner(BaseAssigner):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_riou: float = 1, o2m=False):
        """Creates the matcher
        
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_riou = cost_riou
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.o2m = o2m
        assert cost_class != 0 or cost_bbox != 0 or cost_riou != 0, "all costs cant be 0"

    @torch.no_grad()
    def assign(self, pred_logits, pred_bboxes, gt_labels, gt_bboxes, img_metas):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size*num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size*num_queries, 5] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 5] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], pred_bboxes.shape[0]

        # assign 0 by default
        assigned_gt_inds = pred_bboxes.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        assigned_labels = pred_bboxes.new_full((num_bboxes, ),
            -1,
            dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            min_cost = pred_bboxes.new_ones((num_bboxes, )) * INF
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = pred_bboxes.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, min_cost, labels=assigned_labels)
  
        batch_out_prob = pred_logits.sigmoid() # [num_queries, num_classes]

        tgt_ids = gt_labels  
        tgt_bbox = gt_bboxes
        out_prob = batch_out_prob
        out_bbox = pred_bboxes

        # Compute the classification cost.
        alpha = self.focal_loss_alpha
        gamma = self.focal_loss_gamma
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log()) 
        cost_class = pos_cost_class[:, tgt_ids] 

        out_bbox_ = obb2xyxy(out_bbox,  'le90') #/ image_size_out
        tgt_bbox_ = obb2xyxy(tgt_bbox,  'le90') #/ image_size_out

        # Compute the iou cost betwen boxes
        rotated_iou = rbbox_overlaps(out_bbox, tgt_bbox)
        diou = self.diou(out_bbox_, tgt_bbox_)
        max_iou, _ = torch.max(rotated_iou,dim=1)
        eta = 1.0
        cost_riou = 1.0-(rotated_iou-diou * eta)
        
        C =  self.cost_class * cost_class + self.cost_riou * cost_riou #+ self.cost_bbox * cost_bbox 

        if not self.o2m:
            mincost, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(len(tgt_ids)).to(src_ind)

            # assign all indices to backgrounds first
            assigned_gt_inds[:] = 0
            # assign foregrounds based on matching results
            assigned_gt_inds[src_ind] = tgt_ind + 1
            if gt_labels is not None:
                assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
                pos_inds = torch.nonzero(
                    assigned_gt_inds > 0, as_tuple=False).squeeze()
                if pos_inds.numel() > 0:
                    assigned_labels[pos_inds] = gt_labels[
                        assigned_gt_inds[pos_inds] - 1]
            else:
                assigned_labels = None
            
            return AssignResult(
                num_gt, assigned_gt_inds, max_iou +1., labels=assigned_labels)
        else:
            mincost, src_ind = torch.topk(C, k=6, dim=0, largest=False)
            assigned_gt_inds[:] = 0
            for i, ind in enumerate(src_ind.transpose(0,1)):
                assigned_gt_inds[ind] = i + 1

            if gt_labels is not None:
                assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
                pos_inds = torch.nonzero(
                    assigned_gt_inds > 0, as_tuple=False).squeeze()
                if pos_inds.numel() > 0:
                    assigned_labels[pos_inds] = gt_labels[
                        assigned_gt_inds[pos_inds] - 1]
            else:
                assigned_labels = None
            
            return AssignResult(
                num_gt, assigned_gt_inds, None, labels=assigned_labels)

    def diou(self, bboxes1, bboxes2):
        bboxes1 = bboxes1.reshape([-1, 1, 4])
        center_x1 = (bboxes1[..., 2] + bboxes1[..., 0]) / 2 
        center_y1 = (bboxes1[..., 3] + bboxes1[..., 1]) / 2 
        center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
        center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

        
        out_max_xy = torch.max(bboxes1[..., 2:],bboxes2[:, 2:]) 
        out_min_xy = torch.min(bboxes1[..., :2],bboxes2[:, :2])

        inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
        outer = torch.clamp((out_max_xy - out_min_xy), min=0.00000001)
        outer_diag = (outer[..., 0] ** 2) + (outer[..., 1] ** 2)
        dious = (inter_diag) / outer_diag
        dious = torch.clamp(dious,min=0,max = 1.0)
        return dious
