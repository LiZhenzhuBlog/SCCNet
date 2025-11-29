r""" Evaluate mask prediction """
import torch


class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_mask, batch):
        gt_mask = batch.get('query_mask')

        # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None:
            assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
            query_ignore_idx *= cls.ignore_index
            gt_mask = gt_mask + query_ignore_idx
            pred_mask[gt_mask == cls.ignore_index] = cls.ignore_index

        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [],  [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            _inter = _pred_mask[_pred_mask == _gt_mask]
            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union

def compute_iou_from_stats(area_inter, area_union, eps=1e-6):
    """
    输入:
      - area_inter: Tensor, shape [C] 或 [B, C] 或 [2, C]（与你的实现一致）
      - area_union: Tensor, 同 shape
    返回:
      - miou: float 标量（平均在有样本的类上）
      - iou_per_class: 1D tensor of per-class IoU
    """
    if not torch.is_tensor(area_inter):
        area_inter = torch.tensor(area_inter)
    if not torch.is_tensor(area_union):
        area_union = torch.tensor(area_union)

    # 若有 batch 维度（例如 [B, C]），先在 batch 维度上求和
    if area_inter.dim() == 2:
        inter = area_inter.sum(dim=0).float()
        union = area_union.sum(dim=0).float()
    else:
        inter = area_inter.float()
        union = area_union.float()

    iou_per_class = inter / (union + eps)
    valid = (union > 0).float()
    if valid.sum() > 0:
        miou = (iou_per_class * valid).sum() / valid.sum()
    else:
        miou = torch.tensor(0.0)
    return float(miou), iou_per_class
