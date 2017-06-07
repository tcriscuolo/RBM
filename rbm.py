import sys
import os
import subprocess
import numpy as np
import gzip

import PIL.Image as Image
from utils import load_mnist_data
from utils import tile_raster_images
from utils import sigmoid
from utils import softplus
from numpy.linalg import norm
from tempfile import TemporaryFile

class RBM:
	def __init__(self, input = None, nvisible = None, nhidden = 100, lrate = 0.1,
									W = None, vbias = None, hbias = None, seed = 1234):
		if (input is None):
			try:
				input =  load_mnist_data()[0]
			except:
				print('Not able to load mnist data')

		if (nvisible is None):
			try:
				nvisible = input[0].shape[0]
			except:
				print('Not able to get number of visible units')
				
		''' If not set, the weights are initialized according to
				http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
				where the weights are sampled from a symmetric uniform interval
		'''
		if(W is None):
			t = 4.0 * np.sqrt(6.0 / (nvisible + nhidden))
			W = np.random.uniform(-t, t, (nvisible, nhidden)).astype('float64')

		if(vbias is None):
			vbias = np.zeros(nvisible, dtype = 'float64')

		if(hbias is None):
			hbias = np.zeros(nhidden, dtype = 'float64')
		
		self.nsample = input.shape[0]
		self.imsize = int(np.sqrt(input.shape[1]))

		self.seed = seed
		np.random.seed(seed)

		self.input = input.astype('float64')
		self.nvisible = nvisible
		self.nhidden = nhidden
		self.lrate = lrate
		self.W = W
		self.vbias = vbias
		self.hbias = hbias
	
	''' 
		Simple getters 
	'''
	def get_weights(self):
		return self.W

	def get_vbias(self):
		return self.vbias

	def get_hbias(self):
		return self.hbias

	def get_lrate(self):
		return self.lrate

	def get_data(self):
		return self.input
	
	def get_seed(self):
		return self.seed

	''' 
		upward propagation
		input: state of a visible unit 
		output:

	'''
	def propup(self, vunit):
		dot = np.dot(vunit, self.W)
		if(np.sum(np.isnan(dot)) > 0):
			print('Found the source of evil')
			print(vunit)
			print(self.W)
			input()

		pre_activation = self.hbias + np.dot(vunit, self.W)
		hprob = sigmoid(pre_activation)
		if(np.sum(np.isnan(hprob)) > 0):
			print('WTF hprob', pre_activation, hprob)
		return [pre_activation, hprob]

	'''
		sample a hidden unit given a visible unit configuration
		input: visible unit configuration 

		output:

	'''
	def sample_h_given_v(self, vunit):
		pre_act, hprob = self.propup(vunit)
		hsample = np.random.binomial(n = 1, p = hprob).astype('float64')
		return [pre_act, hprob, hsample]

	def propdown(self, hunit):
		pre_activation = self.vbias + np.dot(hunit, self.W.T)
		vprob = sigmoid(pre_activation)
		return [pre_activation, vprob] 

	'''
			sample visible unit given a hidden unit configuration
			input: hidden unit configuration
			output:

	'''
	def sample_v_given_h(self, hunit):
		pre_act, vprob = self.propdown(hunit)
		vsample = np.random.binomial(n = 1, p = vprob).astype('float64')
		return [pre_act, vprob, vsample]


	''' 
			from a visible unit samples a hidden unit 
			then a visible unit
				
			input:
			output:
	'''
	def gibbs_vhv(self, v0):
		h1pre, h1prob, h1samp = self.sample_h_given_v(v0)
		v1pre, v1prob, v1samp = self.sample_v_given_h(h1samp)
		return [h1pre, h1prob, h1samp, v1pre, v1prob, v1samp]


	'''
			from a hidden unit sample a visible unit
			then  a hidden unit

			intput:

			output:
	'''
	def gibbs_hvh(self, h0):
		v1pre, v1prob, v1samp = self.sample_v_given_h(h0)
		h1pre, h1prob, h1samp = self.sample_h_given_v(v1samp)
		return [v1pre, v1prob, v1samp, h1pre, h1prob, h1samp]

	def free_energy(self, visible):
		left = np.dot(self.vbias, visible)
		right_exp = self.hbias + np.dot(visible, self.W)
		right = np.sum(np.log(1 + np.exp(right_exp)))
		return (-left - right)

	def update_W(self, x, hx, xtilde, hxtilde):
		posup = np.outer(x, hx)
		negup = np.outer(xtilde, hxtilde)
		return (posup - negup)

	def update_vbias(self, x, xtilde):
		return (x - xtilde)

	def update_hbias(self, hx, hxtilde):
		return (hx - hxtilde)

	'''
			Computes the mean stocastic reconstruction error
			of a given input, usually is the input dataset
			but it can be a chunck of it
			input:
						x: data to compute the mean reconstruction error

			output: 
						error: mean reconstruction error of the input x
	'''
	def mean_reconstruction_error(self, x):
		xtilde = self.gibbs_vhv(x)[-1]
		return np.mean(norm(x - xtilde, axis = 1))


	'''
			Computes the mean cross entropy of a 
			given input
			input: 
						x: 
			output: 

	'''
	def mean_cross_entropy(self, x):
		gibs = self.gibbs_vhv(x)
		pre_act = gibs[-2]
		xtilde = gibs[-1]
		ce = np.mean(np.sum(x * np.log(sigmoid(pre_act)) +
						(1 - x)*np.log(1 - sigmoid(pre_act)), axis = 1))
		return ce

	def update_parameters(self, input, xtilde, batch, verbosity = False):

		Wp = np.zeros(self.W.shape, dtype = 'float64')
		hbiasp = np.zeros(self.nhidden, dtype = 'float64')
		vbiasp = np.zeros(self.nvisible, dtype = 'float64')

		batch_size = input.shape[0]
		if(batch_size <= 0):
			print('Kabuuum')
			input()

		for samp in range(batch_size):
			xp = input[samp]
			hxp = self.hbias + np.dot(xp, self.W)
			
			xt = xtilde[samp]
			hxt = self.hbias + np.dot(xt, self.W)
			Wp = Wp + self.update_W(xp, hxp, xt, hxt)/batch_size
			hbiasp = hbiasp + self.update_hbias(hxp, hxt)/batch_size
			vbiasp = vbiasp + self.update_vbias(xp, xt)/batch_size
		
		self.W = self.W + self.lrate * Wp/batch_size
		
		self.hbias = self.hbias + self.lrate*hbiasp/batch_size
		self.vbias = self.vbias + self.lrate*vbiasp/batch_size
	
		if(verbosity):
			print('Wp', norm(Wp))
			print('hbiasp', norm(hbiasp))
			print('vbiasp', norm(vbiasp))
			print('mean rerror', self.mean_reconstruction_error(input))
			print('mean entropy', self.mean_cross_entropy(input))

	def perform_cd(self, input, k):
		xtilde = input
		for kprime in range(k):
			xtilde = self.gibbs_vhv(xtilde)[-1]	
		return xtilde

	def constrative_divergence(self, nepochs = 10, batch_size = 30, cdk = 5, epoch_stats = True, auto_save = True):
		nbatches = self.input.shape[0] // batch_size
		print('-- Training using constrative divergence')
		print('nhidden %d' % self.nhidden)
		print('lrate %f' % self.lrate)
		print('nepochs: %d' % nepochs)
		print('bsize: %d' % batch_size)
		print('k: %d' % cdk)

		for epoch in range(nepochs):
			for batch in range(nbatches):

				x = self.input[batch*batch_size:(batch+1)*batch_size]
				xtilde = self.perform_cd(x, cdk)
				this_batch_size = x.shape[0]
				self.update_parameters(x, xtilde, batch)
			
		
			if(epoch_stats):
				print('Epoch %d stats' % (epoch))
				print('\t\tMean Reconstruction Error %f' % self.mean_reconstruction_error(self.input))
				print('\t\tMean Cross Entropy Error %f' % self.mean_cross_entropy(self.input))
			
			if(auto_save):
				self.save_model('k_%d_cd_epoch_%d.model' % (cdk, epoch))	

			self.plot_hidden_units('plots/k_%d_cd_filters_epoch_%d.png' % (cdk, epoch))
	
	def plot_hidden_units(self, path, tile_shape = (10, 10)):
		imsize = self.imsize
		tiled_image = tile_raster_images(X = self.W.T, img_shape = (imsize	, imsize),
										tile_shape = tile_shape, tile_spacing=(1, 1))
		image = Image.fromarray(tiled_image)
		image.save(path)


	'''
			Save the RBM model in a specified file 
			input: 
						output: name of the output file
						dir_path: directory to save the output file

	'''
	def save_model(self, output, dir_path = 'models_param'):
		
		if not os.path.exists(dir_path):
			try:
				os.makedirs(dir_path)
			except:
				print('Could not make directory')
				raise	
		
		file = dir_path + '/' + output
		
		try:	
			with open(file, 'wb') as out:

				np.savez(out, input = self.input,
										constants = [self.nvisible, self.nhidden, self.lrate, self.seed],
										weights = self.W,
										vbias = self.vbias, 
										hbias = self.hbias)
		
		except:
			print('Could not save in %s', file)
			raise


	'''
			Load the model from a specified file
			input: 
						input: name of the file to load the RBM model
						dir_path: directory where the model input file is
	'''
	def load_model(self, input, dir_path = 'models_param'):
	
		try:
			file = dir_path + '/' + input
			npfile = np.load(file)
		
			print('--- Loading data from %s' % input)
			self.input = npfile['input']
			self.W = npfile['weights']
			self.vbias = npfile['vbias']
			self.hbias = npfile['hbias']
		
			self.nvisible, self.nhidden, self.lrate, self.seed = npfile['constants']
			self.nvisible = int(self.nvisible)
			self.nhidden = int(self.nhidden)
			self.seed = int(self.seed)
			self.nsample = self.input.shape[0]
			self.imsize = int(np.sqrt(self.input.shape[1]))
		except:
			raise
	
	def sample_from_model(self, samples_to_take = 10, number_of_chains = 10, k = 1):
		selected_idx = np.random.choice(range(self.nsample), samples_to_take)
		digits = self.input[selected_idx]
		
		tile_shape = (samples_to_take, number_of_chains)
			
		shape = (samples_to_take * number_of_chains, self.imsize * self.imsize) 
		images = np.zeros(shape)
		for idx in range(samples_to_take):
			images[idx] = digits[idx] 
		
		for chain in range(1, number_of_chains):
			digits = self.perform_cd(digits, k)
			for idx in range(samples_to_take):
				images[chain*samples_to_take + idx] = digits[idx] 
		


		return tile_raster_images(images, (self.imsize, self.imsize), (samples_to_take, number_of_chains))
