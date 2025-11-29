r""" SCCNetwork Implementation"""
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner


class SCCNetwork(nn.Module):
    def __init__(self, backbone, use_original_imgsize, freeze=True):
        super(SCCNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        if freeze:
            self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.hpn_learner2 = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.merger = nn.Sequential(nn.Conv2d(4, 2, (1, 1), bias=False), nn.ReLU())

    def forward(self, query_img, support_img, support_mask, query_mask=None):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.mask_feature(support_feats, support_mask.clone())
            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
        
        logit_mask_ori = self.hpn_learner(corr)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask_ori, support_img.size()[2:], mode='bilinear', align_corners=True)
        
        pred_mask = logit_mask.argmax(dim=1)
        with torch.no_grad():
            masked_qfeats = self.mask_feature(query_feats, pred_mask)
            corr2 = Correlation.multilayer_correlation(query_feats, masked_qfeats, self.stack_ids)

        logit_mask2 = self.hpn_learner2(corr2)
        logit = torch.cat([logit_mask_ori, logit_mask2], dim=1)
        logit = self.merger(logit)
        if not self.use_original_imgsize:
            logit = F.interpolate(logit, support_img.size()[2:], mode='bilinear', align_corners=True)
        
        logit_mask3 = None
        if query_mask is not None:
            with torch.no_grad():
                masked_qfeats3 = self.mask_feature(query_feats, query_mask)
                corr3 = Correlation.multilayer_correlation(query_feats, masked_qfeats3, self.stack_ids)
            logit_mask3 = self.hpn_learner2(corr3)
            
            if not self.use_original_imgsize:
                logit_mask3 = F.interpolate(logit_mask3, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit, logit_mask3

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):
    
        # 初始化聚合张量为 Softmax 概率 (浮点数)
        # 我们将聚合 Softmax 后的前景概率 (通道 1)
        
        # 临时变量，用于存储第一个 Softmax 结果的大小，以便初始化 agg
        with torch.no_grad():
            logit_mask, _ = self(batch['query_img'], batch['support_imgs'][:, 0], batch['support_masks'][:, 0])
            # 使用 Softmax 获取前景概率
            probs = F.softmax(logit_mask, dim=1)[:, 1, ...].clone() # [B, H, W]
        
        # 初始化聚合 Softmax 概率 (只聚合前景通道)
        # logit_mask_agg 现在存储的是前景概率的总和
        prob_mask_agg = probs
        
        if nshot > 1:
            # Loop over remaining shots
            for s_idx in range(1, nshot): # 从 1 开始循环
                with torch.no_grad():
                    logit_mask, _ = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
                    probs = F.softmax(logit_mask, dim=1)[:, 1, ...].clone() # [B, H, W]
                
                prob_mask_agg += probs # 聚合 Softmax 概率
                
        # ---------------- 最终量化 ----------------
        
        # 1. 计算平均 Softmax 概率
        # pred_mask 是平均概率 [B, H, W]
        pred_mask = prob_mask_agg / nshot 
        
        # 2. Rescale to Original Size (如果需要)
        if self.use_original_imgsize:
            # 注意: pred_mask 现在是 [B, H, W]，需要 unsqueeze(1) 才能用 F.interpolate
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            pred_mask = F.interpolate(
                pred_mask.unsqueeze(1), 
                org_qry_imsize, 
                mode='bilinear', 
                align_corners=True
            ).squeeze(1) # 恢复到 [B, H, W]
        
        # 3. 硬性二值化 (使用 0.5 作为标准阈值)
        # 移除 0.4 的硬性阈值，使用更标准的 0.5
        threshold = 0.5 
        pred_mask = (pred_mask >= threshold).long()
    
        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def focal_loss(self, x, p=1, c=0.1):
        return -torch.pow(1 - x, p) * torch.log(c + x)

    def compute_area_loss(self, pred_mask, gt_mask):
        pred_area = pred_mask.flatten().float().mean()
        gt_area = gt_mask.flatten().float().mean()
        ratio = torch.minimum(pred_area, gt_area) / (0.01 + torch.maximum(pred_area, gt_area))
        return self.focal_loss(ratio)

    def compute_focal_loss(self, pred_mask):
        pred_mask = pred_mask.flatten().float()
        return self.focal_loss(pred_mask.mean())

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
