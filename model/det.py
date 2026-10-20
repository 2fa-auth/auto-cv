#!/home/client/Documents/fun/venv/bin/python3
from os import path, listdir, cpu_count
from PIL import Image 

import torch
from torch.optim import AdamW
from torchvision.tv_tensors import BoundingBoxes
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as V2
import torch.nn as nn

from loss import BboxLoss_withMSE
from utils import denorm_coords, norm_coords, detection_objects, view_image
from model import FastResidualDetector
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
    
    for label in self.lbs_list:
      if label.split('.')[0] == target_img.split('.')[0]:
        with open(path.join(self.labels, label), "r") as f:
          for t in list(filter(None, f.read().split('\n'))):
            plbl = list(map(float, t.split(' '))) 
            for i in range(len(plbl)-1): 
              if plbl[i+1] > 1.0: plbl = 0; break
            if plbl and plbl[0] == 0.: labels.append(plbl)
    
    labels = torch.tensor(labels)
    image = Image.open(path.join(self.images,target_img)).convert('RGB')

    if self.transforms:
      bbox = BoundingBoxes(
        data=denorm_coords(labels[:, 1:]), 
        format='CXCYWH', 
        canvas_size=config.IMAGE_SIZE
      )
      image, labels[:, 1:] = self.transforms(image, bbox)
      labels[:, 1:] = norm_coords(bbox)

    class_ids = labels[:, 0].to(torch.int32)
    labels=torch.cat([
      labels,
      torch.ones(labels.size(0), 1),
      torch.zeros_like(labels),
      torch.zeros(labels.size(0),
      config.C)], dim=1)[:, 1:]
    labels[..., 10+class_ids]=1 # [N, P * B + C]
    
    target = torch.zeros(config.S, config.S, 5 * config.B + config.C)

    cell_x = torch.floor(labels[:, 0] * config.S).to(torch.int32)
    cell_y = torch.floor(labels[:, 1] * config.S).to(torch.int32)

    target[cell_y, cell_x] = labels

    # for i in range(target.size(0)): # [i, S, P]
      # for j in range(target.size(1)): # [S, j, P]
        # print(target[i, j, :])
      # print()

    return image, target


class ModelTrainer(nn.Module):
  def __init__(self, model, train_set, val_set, test_set):
    super().__init__()

    self.dev = 'cuda' if torch.cuda.is_available else 'cpu'
    self.model = model.to(self.dev)
    
    num_processors = cpu_count()
    self.train_loader = DataLoader(
      dataset=train_set, 
      batch_size=config.BATCH_SIZE, 
      shuffle=False, 
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
      data = data.to(self.dev, non_blocking=True)
      labels = labels.to(self.dev, non_blocking=True)
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

        # print(f'labels 1:\n{labels[0][1]}')
        # print(f'labels 2:\n{labels[0][2]}\n')
        
        # print(f'pred 1:\n{predictions[0][1]}')
        # print(f'pred 2:\n{predictions[0][2]}\n')
        # x = detection_objects(data[0], predictions[0]) 
        # view_image('asd', x)
        # break

    torch.save(self.model.state_dict(), 'weights.pth')
    return losses / len(self.test_loader)

  def forward(self):
    # for x, y in self.train_loader: # отобразить изображение с рамками
      # x = detection_objects(x[0], y[0])
      # view_image('image', x)
      # exit()  
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
  transforms = V2.Compose([
    V2.Resize(config.IMAGE_SIZE),
    V2.ToImage(),
    V2.ToDtype(torch.float32, scale=True),
  ])

  trainer = ModelTrainer(
    model = FastResidualDetector(
      in_chan=3,
      out_features=5 * config.B + config.C
    ),
    train_set=SetNoob(config.PATH_TO_DATASET, config.NAME_DATASET, train=True, transforms=transforms),
    val_set=SetNoob(config.PATH_TO_DATASET, config.NAME_DATASET, train=True, transforms=transforms),
    test_set=SetNoob(config.PATH_TO_DATASET, config.NAME_DATASET, train=True, transforms=transforms)); 
  trainer()
  print(f'общий показатель ошибки модели: {trainer.test()}')