#!/home/client/Documents/fun/venv/bin/python3
import os
from PIL import Image 
import torch
from torch.optim import AdamW
from torchvision.tv_tensors import BoundingBoxes
import torch.utils.data as data
import torchvision.transforms.v2 as v2
import torch.nn as nn

from utils import denorm_coords, norm_coords, get_iou, detection_objects
from backbone import BACKBONEModel
import config

class SetNoob(data.Dataset):
  def __init__(self, pathto_dset, namedset, train=True, transforms=None):
    self.images = os.path.join(pathto_dset, namedset, "images", "train" if train else "val")
    self.labels = os.path.join(pathto_dset, namedset, "labels", "train" if train else "val")
    self.imgs_list = os.listdir(self.images)
    self.lbs_list = os.listdir(self.labels)
    self.transforms = transforms

  def __len__(self): 
    return len(self.imgs_list)    
  
  def __getitem__(self, index):
    image=self.imgs_list[index]
    labels=[]
    # парсинг label*.txt файла
    for label in self.lbs_list:
      if label.split('.')[0] == image.split('.')[0]:
        with open(os.path.join(self.labels, label), "r") as f:
          ts = f.read().split('\n') 
          for t in ts: 
            ar = []
            for val in t.split(' '): 
              if val != '': ar.append(float(val))
            if ar: labels.append(ar)

    labels = torch.tensor(labels)
    image = Image.open(os.path.join(self.images,image)).convert('RGB')
    w, h = image.size

    if self.transforms:
      bbox = BoundingBoxes(data=denorm_coords(labels[:, 1:]), format='CXCYWH', canvas_size=(h,w))
      image, labels[:, 1:] = self.transforms(image, bbox)
      labels[:, 1:] = norm_coords(bbox)

    # подготовка меток
    class_ids = labels[:, 0].to(torch.int32)
    labels=torch.cat([
      labels,
      torch.ones(labels.size(0), 1),
      torch.zeros_like(labels),
      torch.zeros(labels.size(0),
      config.C)], dim=1)[:, 1:]
    for i, ax in enumerate(labels): ax[10+class_ids[i]] = 1

    # создание сетки H_grid, W_grid и запись в оси X и Y сетки 
    target=torch.zeros(config.S, config.S, 5 * config.B + config.C)
    cell_x = torch.floor(labels[:, 0] * config.S).to(torch.int32) 
    cell_y = torch.floor(labels[:, 1] * config.S).to(torch.int32)
    target[cell_x, cell_y] = labels

    return image, target

class BboxLoss_withMSE(nn.Module):
  def __init__(self): 
    self.l_coord = 5
    self.l_nooobj = 0.5
    super().__init__()

  def forward(self, p, y):
    iou_b1 = get_iou(p[..., :4], y[..., :4])
    iou_b2 = get_iou(p[..., 5:9], y[..., :4])
    """
    nomain_box = main_box.detach().clone()
    for b in range(iou_b1.size(0)):
      for s1 in range(iou_b1.size(1)):
        for s2 in range(iou_b1.size(2)):
          for v in range(iou_b1.size(3)):
            if iou_b1[b, s1, s2, v] > iou_b2[b, s1, s2, v]:
              main_box[b, s1, s2, :5] = p[b, s1, s2, :5] # bbox1 - main
              nomain_box[b, s1, s2, :5] = p[b, s1, s2, 5:10] # bbox2 - no main
            elif iou_b1[b, s1, s2, v] <= iou_b2[b, s1, s2, v]: 
              main_box[b, s1, s2, :5] = p[b, s1, s2, 5:10] # bbox2 - main
              nomain_box[b, s1, s2, :5] = p[b, s1, s2, :5] # bbox1 - no main
    """
    main_box = torch.where(iou_b1 <= iou_b2, p[..., 5:10], p[..., :5])
    nomain_box = torch.where(iou_b1 <= iou_b2, p[..., :5], p[..., 5:10])

    mse_loss = nn.MSELoss(reduction='sum')

    #XY
    x_loss = mse_loss(
      main_box[..., 0:1], 
      y[..., 0:1]
    )
    y_loss = mse_loss(
      main_box[..., 1:2], 
      y[..., 1:2]
    )
    xy_loss = x_loss + y_loss

    #WH
    w_loss = mse_loss(
      main_box[..., 2:3], 
      y[..., 2:3]
    )
    h_loss = mse_loss(
      main_box[..., 3:4], 
      y[..., 3:4]
    )
    wh_loss = w_loss + h_loss
    coord_loss = xy_loss + wh_loss 

    #obj confidence
    obj_confidence_losse = mse_loss(
      main_box[..., 4:5], 
      torch.ones_like(iou_b1)
    )

    #no obg confidence
    noobj_confidence_losse = mse_loss(
      nomain_box[..., 4:5], 
      torch.zeros_like(iou_b2)
    )

    #classification
    cross = nn.CrossEntropyLoss(reduction='sum')
    class_loss = cross(p[..., 10:config.C], y[..., 10:config.C])

    return self.l_coord * coord_loss \
        + obj_confidence_losse \
        + self.l_nooobj * noobj_confidence_losse \
        + class_loss


class ModelTrainer(nn.Module):
  def __init__(self, model, train_set, val_set, test_set):
    super().__init__()
    self.dev = 'cuda' if torch.cuda.is_available else 'cpu'
    self.model = model.to(self.dev)
    self.train_loader = data.DataLoader(dataset=train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    self.val_loader = data.DataLoader(dataset=val_set, batch_size=config.BATCH_SIZE, shuffle=False)  
    self.test_loader = data.DataLoader(dataset=test_set, batch_size=config.BATCH_SIZE, shuffle=True)
    self.criterion = BboxLoss_withMSE()
    self.optimizer = AdamW(params=self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
  
  def train(self):
    losses = 0
    for data, labels in self.train_loader:
      data = data.to(self.dev)
      labels = labels.to(self.dev)
      predictions = self.model(data)
      loss = self.criterion(predictions, labels)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      losses += loss.item()
      
      del data, labels
    return losses / len(self.train_loader)

  def valid(self):
    losses = 0
    with torch.no_grad():
      for data, labels in self.val_loader:
        data = data.to(self.dev)
        labels = labels.to(self.dev)
        predictions = self.model(data)
        losses += self.criterion(predictions, labels).item()
        
        del data, labels
    return losses / len(self.val_loader)
  
  def test(self):
    losses = 0
    with torch.no_grad():
      for data ,labels in self.test_loader:
        data = data.to(self.dev)
        labels = labels.to(self.dev)
        predictions = self.model(data)
        losses += self.criterion(predictions, labels).item()

        # detection_objects(data[0], predictions[0]) 
        break

    torch.save(self.model.state_dict(), 'weights.pth')
    
    return losses / len(self.test_loader)

  def forward(self):
    #for x, y in self.train_loader: # отобразить изображение с рамками
      #x = utils.detection_objects(x[0], y[0])
      #utils.view_image('image', x)
      #exit()  
    self.model.train()
    for _ep in range(config.EPOCHS):
      # self.model.train()
      train_meanloss = self.train()
      # self.model.eval()
      # val_meanloss = self.valid()
      # print(f'[{_ep}/{EPOCHS}]\tLOSS TRAIN {train_meanloss}\tLOSS VAL {val_meanloss}')
      print(f'[{_ep}/{config.EPOCHS}]\tLOSS {train_meanloss}')
    self.model.eval()


if __name__ == "__main__":
  transforms = v2.Compose([
    v2.Resize(config.IMAGE_SIZE),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
  ])

  trainer = ModelTrainer(
    model=BACKBONEModel(3, 5 * config.B + config.C), 
    train_set=SetNoob("./", "kitti",train=True, transforms=transforms),
    val_set=SetNoob("./", "kitti",train=False, transforms=transforms),
    test_set=SetNoob("./", "kitti",train=False, transforms=transforms)); 
  trainer()
  print(f'общий показатель ошибки модели: {trainer.test()}')