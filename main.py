
from rbm import RBM
import numpy as np
import PIL.Image as Image

if __name__ == '__main__':

	M = RBM()
	M.constrative_divergence()
	m = np.asarray([[1, 2, 3, 4], [2, 3, 4, 5], [2, 3, 1, 1]])

