#!/home/client/Documents/fun/venv/bin/python3
from os import path, listdir, cpu_count
from PIL import Image 
import torch
from torch.optim import AdamW
from torchvision.tv_tensors import BoundingBoxes
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import torch.nn as nn
from utils import (
    denorm_coords, 
    norm_coords, 
    get_iou, 
    detection_objects, 
    view_image,
  )
from backbone import BACKBONEModel
import config

class SetNoob(Dataset):
  def __init__(self, pathto_dset, namedset, train=True, transforms=None):
    self.images = path.join(pathto_dset, namedset, "images", "train" if train else "val")
    self.labels = path.join(pathto_dset, namedset, "labels", "train" if train else "val")
    self.imgs_list = listdir(self.images)
    self.lbs_list = listdir(self.labels)
    self.transforms = transforms
  
  def __len__(self): 
    return len(self.imgs_list)
  
  def __getitem__(self, index):
    target_img=self.imgs_list[index]
    labels=[]
    # нахождение соответствующего label файл и парсинг
    for label in self.lbs_list:
      if label.split('.')[0] == target_img.split('.')[0]:
        with open(path.join(self.labels, label), "r") as f:
          for t in list(filter(None, f.read().split('\n'))):
            plbl = list(map(float, t.split(' '))) 
            for i in range(len(plbl)-1): 
              if plbl[i+1] > 1.0: plbl = 0; break
            if plbl and plbl[0] == 0.: labels.append(plbl)
    # трансформация 
    labels = torch.tensor(labels)
    image = Image.open(path.join(self.images,target_img)).convert('RGB')
    if self.transforms:
      bbox = BoundingBoxes(
        data=denorm_coords(labels[:, 1:]), 
        format='CXCYWH', 
        canvas_size=tuple(reversed(image.size))
      )
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

    # создание сетки H_grid, W_grid и запись в оси  
    target=torch.zeros(config.S, config.S, 5 * config.B + config.C)
    cell_x = torch.floor(labels[:, 0] * config.S).to(torch.int32) 
    cell_y = torch.floor(labels[:, 1] * config.S).to(torch.int32)

    target[cell_x, cell_y] = labels

    return image, target

class BboxLoss_withMSE(nn.Module):
  def __init__(self): 
    super().__init__()
    self.l_coord = 5
    self.l_nooobj = 0.5

  def forward(self, p, y):
    iou_b1 = get_iou(p[..., :4], y[..., :4])
    iou_b2 = get_iou(p[..., 5:9], y[..., :4])

    main_box = torch.where(iou_b1 <= iou_b2, p[..., 5:10], p[..., :5])
    nomain_box = torch.where(iou_b1 <= iou_b2, p[..., :5], p[..., 5:10])

    mask = y[..., 4:5]
    x_loss = (
      (mask * (main_box[..., 0:1]-y[..., 0:1])).pow(2)
    ).sum()
    y_loss = (
      (mask * (main_box[..., 1:2]-y[..., 1:2])).pow(2)
    ).sum()
    w_loss = (
      (mask * (main_box[..., 2:3]-y[..., 2:3])).pow(2)
    ).sum()
    h_loss = (
      (mask * (main_box[..., 3:4]-y[..., 3:4])).pow(2)
    ).sum()
    coord_loss = x_loss + y_loss + w_loss + h_loss 

    obj_confidence_loss = (
      mask * (main_box[..., 4:5] - get_iou(main_box[..., :4], y[..., :4]).detach()).pow(2)
    ).sum()

    noobj_confidence_loss = (
      mask * nomain_box[..., 4:5].pow(2)
      + (1 - mask) * main_box[..., 4:5].pow(2)
      + (1 - mask) * nomain_box[..., 4:5].pow(2)
    ).sum()

    # классификация
    cross = nn.CrossEntropyLoss(reduction='none')
    class_loss = ((
      cross(
        p[..., 10:10+config.C].reshape(-1, config.C), 
        y[..., 10:10+config.C].reshape(-1, config.C)
        )) * mask.reshape(-1)
    ).sum()

    return self.l_coord * coord_loss \
      + obj_confidence_loss \
      + self.l_nooobj * noobj_confidence_loss \
      + class_loss


class ModelTrainer(nn.Module):
  def __init__(self, model, train_set, val_set, test_set):
    super().__init__()
    self.dev = 'cuda' if torch.cuda.is_available else 'cpu'
    self.model = model.to(self.dev)
    
    num_processors = cpu_count() # ядер процессора
    self.train_loader = DataLoader(
      dataset=train_set, 
      batch_size=config.BATCH_SIZE, 
      shuffle=True, 
      num_workers=num_processors,
      pin_memory=True,
      persistent_workers=True,
    )
    self.val_loader = DataLoader(
      dataset=val_set, 
      batch_size=config.BATCH_SIZE, 
      shuffle=False,
      num_workers=num_processors,
      pin_memory=True,
      persistent_workers=True
    )  
    self.test_loader = DataLoader(
      dataset=test_set,
      batch_size=config.BATCH_SIZE, 
      shuffle=True,
      num_workers=num_processors,
      pin_memory=True,
      persistent_workers=True,
    )
    self.criterion = BboxLoss_withMSE()
    self.optimizer = AdamW(params=self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
  
  def train(self):
    losses = 0
    for data, labels in self.train_loader:
      # благодаря pin_memory, перенос (.to) сработает максимально быстро
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
    for x, y in self.train_loader: # отобразить изображение с рамками
      x = detection_objects(x[0], y[0])
      view_image('image', x)
      exit()  
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
    train_set=SetNoob(config.PATH_TO_DATASET, config.NAME_DATASET,train=True, transforms=transforms),
    val_set=SetNoob(config.PATH_TO_DATASET, config.NAME_DATASET,train=False, transforms=transforms),
    test_set=SetNoob(config.PATH_TO_DATASET, config.NAME_DATASET,train=False, transforms=transforms)); 
  trainer()
  print(f'общий показатель ошибки модели: {trainer.test()}')