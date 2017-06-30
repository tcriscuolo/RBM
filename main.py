
from rbm import RBM
from utils import load_mnist_data
import numpy as np
import PIL.Image as Image

if __name__ == '__main__':
	data = load_mnist_data()
	train_x, train_y, valid_x, valid_y, test_x, test_y = data

	print(train_x.shape)
	print(valid_x.shape)
	print(test_x.shape)

	M = RBM()
	#M.contrastive_divergence(validation = valid_x)
	#m = np.asarray([[1, 2, 3, 4], [2, 3, 4, 5], [2, 3, 1, 1]])

