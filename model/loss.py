import torch
from torch.nn import Module, CrossEntropyLoss
from config import C
from utils import get_iou

class BboxLoss_withMSE(Module):
  def __init__(self): 
    super().__init__()
    self.cl,self.nc=(5,0.5)
  
  def forward(self, p, y):
    mask = y[..., 4:5]

    iou_b1 = get_iou(p[..., :4], y[..., :4]) # [B, S, S, 1]
    iou_b2 = get_iou(p[..., 5:9], y[..., :4]) # [B, S, S, 1]

    main_box = torch.where(iou_b1 <= iou_b2, p[..., 5:10], p[..., :5]) # [B, S, S, 4]
    nomain_box = torch.where(iou_b1 <= iou_b2, p[..., :5], p[..., 5:10]) # [B, S, S, 4]

    x_loss = ((mask * (main_box[..., 0:1]-y[..., 0:1])).pow(2)).sum()
    y_loss = ((mask * (main_box[..., 1:2]-y[..., 1:2])).pow(2)).sum()
    w_loss = ((mask * (main_box[..., 2:3]-y[..., 2:3])).pow(2)).sum()
    h_loss = ((mask * (main_box[..., 3:4]-y[..., 3:4])).pow(2)).sum()
    coord_loss = (x_loss+y_loss+w_loss+h_loss) / mask.sum().clamp_min(1) 

    obj_confidence_loss = (mask * (main_box[..., 4:5] - get_iou(main_box[..., :4], y[..., :4]).detach()).pow(2)).sum()
    noobj_confidence_loss = (
      mask * nomain_box[..., 4:5].pow(2)
      + (1 - mask) * main_box[..., 4:5].pow(2)
      + (1 - mask) * nomain_box[..., 4:5].pow(2)
    ).sum() / mask.sum().clamp_min(1)

    criterion = CrossEntropyLoss(reduction='none')
    class_loss = ((
      criterion(p[..., 10:10+C].reshape(-1, C), y[..., 10:10+C].reshape(-1, C))) * mask.reshape(-1)
    ).sum() / (1 - mask).sum().clamp_min(1)

    return self.cl*coord_loss + obj_confidence_loss + self.nc*noobj_confidence_loss + class_loss