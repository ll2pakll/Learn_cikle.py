from Help_fn.torch_model import *
import pickle
from matplotlib import pyplot as plt

epochs = -1 # с какой эпохи начинать рисовать график

def load_picle_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

spisok_loss = load_picle_file(spisok_loss_path)
plt.figure(figsize=(40, 40))
plt.plot(spisok_loss[-1*epochs:])
plt.show()