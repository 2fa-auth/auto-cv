#!/home/client/Documents/fun/venv/bin/python3
import torch
import cv2

def IoU(bbox1, bbox2):
  box1 = cxcywh_to_xyxy(bbox1)
  box2 = cxcywh_to_xyxy(bbox2)

  x1_box1 = torch.min(box1[..., 0:1], box1[..., 2:3])
  y1_box1 = torch.min(box1[..., 1:2], box1[..., 3:4]) 
  x2_box1 = torch.max(box1[..., 2:3], box1[..., 0:1])
  y2_box1 = torch.max(box1[..., 3:4], box1[..., 1:2])
  x1_box2 = torch.min(box2[..., 0:1], box2[..., 2:3])
  y1_box2 = torch.min(box2[..., 1:2], box2[..., 3:4]) 
  x2_box2 = torch.max(box2[..., 2:3], box2[..., 0:1])
  y2_box2 = torch.max(box2[..., 3:4], box2[..., 1:2])
  x1_box = torch.max(x1_box1, x1_box2)
  y1_box = torch.max(y1_box1, y1_box2)
  x2_box = torch.min(x2_box1, x2_box2)
  y2_box = torch.min(y2_box1, y2_box2)

  intersection_area = torch.clamp(x2_box - x1_box, 0) * torch.clamp(y2_box - y1_box, 0)
  wbox1 = x2_box1 - x1_box1
  hbox1 = y2_box1 - y1_box1
  wbox2 = x2_box2 - x1_box2 
  hbox2 = y2_box2 - y1_box2
  area_box1 = wbox1 * hbox1
  area_box2 = wbox2 * hbox2
  union_area = area_box1 + area_box2 - intersection_area
  return intersection_area / (union_area + 1e-6)

def denorm_coords(bbox, w, h): 
  return bbox * torch.tensor([w, h, w, h])

def norm_coords(bbox, w, h): 
  return bbox / torch.tensor([w, h, w, h])

def cxcywh_to_xyxy(coords):
  coords = coords.clone()
  x1 = coords[..., 0] - coords[..., 2] / 2
  y1 = coords[..., 1] - coords[..., 3] / 2
  x2 = coords[..., 0] + coords[..., 2] / 2
  y2 = coords[..., 1] + coords[..., 3] / 2
  coords[..., 0] = x1
  coords[..., 1] = y1
  coords[..., 2] = x2
  coords[..., 3] = y2 
  return coords

def extract_labels(grid, bbox=True): 
  tlist=[]
  for ax1 in grid:
    for preds in ax1:
      if torch.round(preds[4]) <= 0: continue # индекс 4 - confidence
      else: tlist.append((preds[:4] if bbox else preds))

  if tlist: return torch.stack([t for t in tlist if t.numel() > 0], 0)
  else: return torch.tensor([])

def view_image(name_img, img):
  cv2.imshow(name_img, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def detection_objects(x, y, own_set_frame=False): 
  import det
  if not own_set_frame:
    x = x.to('cpu')
    x = cv2.cvtColor(x.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
  
  bboxes = extract_labels(y).to('cpu') # [H_grid, W_grid, preds]
  if bboxes.numel() == 0: return x 
  bboxes = cxcywh_to_xyxy(bboxes)
  bboxes = denorm_coords(bboxes, det.W_image, det.H_image)

  for bbox in bboxes:  
    x_min = int(bbox[0].item()) 
    y_min = int(bbox[1].item()) 
    x_max = int(bbox[2].item()) 
    y_max = int(bbox[3].item()) 
    cv2.rectangle(x, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
  return x

# сортирует bbox по x1 
def sort_by_x1(target):
  indices = torch.argsort(target[..., 0])
  return torch.gather(input=target, dim=1, index=indices.unsqueeze(-1).expand_as(target))