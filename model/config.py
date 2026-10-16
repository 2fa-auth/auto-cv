NAME_DATASET = "kitti"
PATH_TO_DATASET = "./"

BATCH_SIZE = 1
EPOCHS = 2
LEARNING_RATE = 1E-4

IMAGE_SIZE = (448, 448) # все изображения приравниваются к единому размеру

S = 7 # изображение в SxS сетке
B = 2 # число ограничивающих рамок в предсказании
C = 8 # количество классов в детесете