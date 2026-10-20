NAME_DATASET = "Intersection-Flow-5K"
PATH_TO_DATASET = "./"

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1E-4

IMAGE_SIZE = (448, 448) # все изображения приравниваются к единому размеру

S = 7 # изображение в SxS сетке
B = 2 # число ограничивающих рамок в предсказании
C = 1 # количество классов в детесете