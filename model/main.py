#!/home/client/Documents/fun/venv/bin/python3
import cv2
from backbone import BACKBONEModel
from utils import detection_objects

import torch
import numpy as np
import torchvision.transforms.v2 as v2


def main():
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = BACKBONEModel(3, 18).to(dev)
  model.load_state_dict(torch.load('weights.pth', map_location=dev, weights_only=True))

  trans = v2.ToDtype(torch.float32, scale=True)
  cap = cv2.VideoCapture('mp4/drive_test1.mp4')

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)   
    frame = trans(torch.tensor(np.expand_dims(frame, axis=0).transpose(0, 3, 1, 2)))
    
    pred = model(frame.to(dev)).to('cpu').squeeze()
    frame = cv2.cvtColor(frame.permute(0, 2, 3, 1).squeeze().detach().numpy(), cv2.COLOR_RGB2BGR)

    img_bbox = detection_objects(frame, pred, own_set_frame=True)
    cv2.imshow('image', img_bbox)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()