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
									W = None, vbias = None, hbias = None, seed = 123):
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
			t = 4.0 * np.sqrt(6.0/float(nvisible + nhidden))
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
		pre_activation = self.hbias + np.dot(vunit, self.W)
		hprob = sigmoid(pre_activation)
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
	def gibbs_vhv(self, v0samp):
		h1pre, h1prob, h1samp = self.sample_h_given_v(v0samp)
		v1pre, v1prob, v1samp = self.sample_v_given_h(h1samp)
		return [h1pre, h1prob, h1samp, v1pre, v1prob, v1samp]

	def free_energy(self, visible):
		left = np.dot(visible, self.vbias)
		right_exp = self.hbias + np.dot(visible, self.W)
		right = np.sum(np.log(1.0 + np.exp(right_exp)), axis = 1)
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
		gibbs = self.gibbs_vhv(x)
		xprob = gibbs[-2]
		xtilde = gibbs[-1]
		ce = np.mean(np.sum(x * np.log(xprob) +
						(1.0 - x)*np.log(1.0 - xprob), axis = 1))
		return ce


	''' 
			Computes the mean pseudo likelihood flipping only a single bit 
			of the visible unit

			input: 
						x: input to compute mean pseudo likelihood
						idx: index of the bit to flip
			output:
						 an estimative of the input x 
						 mean pseudo likelihood
	'''
	def mean_pseudo_likelihood(self, x, idx):
		xfe = self.free_energy(x)
		# swaps the ith bit of each image
		means = []
		for index in idx:
			# flip the ith visible unit
			x[: , index] = 1 - x[:, index]	
			nxfe = self.free_energy(x)
			# flip back the ith visible unit
			x[:, index] = 1 - x[:, index]
			means.append(self.nvisible * np.log(sigmoid(nxfe - xfe)))
		return np.mean(means)
	


	def update_parameters(self, input, xtilde, verbosity = False):

		Wp = np.zeros(self.W.shape, dtype = 'float64')
		hbiasp = np.zeros(self.nhidden, dtype = 'float64')
		vbiasp = np.zeros(self.nvisible, dtype = 'float64')

		batch_size = input.shape[0]
		
		for samp in range(batch_size):
			xp = input[samp]
			hxp = sigmoid(self.hbias + np.dot(xp, self.W))
			xt = xtilde[samp]
			hxt = sigmoid(self.hbias + np.dot(xt, self.W))
			Wp += + self.update_W(xp, hxp, xt, hxt)
			hbiasp += self.update_hbias(hxp, hxt)
			vbiasp += self.update_vbias(xp, xt)
		
		blambda = self.lrate/float(batch_size)
		self.W = self. W + (blambda * Wp)
		self.hbias = self.hbias + (blambda * hbiasp)
		self.vbias = self.vbias + (blambda * vbiasp)
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

	def validate_model(self, validation, epoch, size = 40):
		print('Epoch %d stats' % (epoch))
		print('\t\tMean Reconstruction Error %f' % self.mean_reconstruction_error(validation))
		print('\t\tMean Cross Entropy Error %f' % self.mean_cross_entropy(validation))
		bit_index = np.random.choice(a = validation.shape[1], size = size)
		print('\t\tMean pseudo likelihood error %f' % self.mean_pseudo_likelihood(validation, bit_index))

	def contrastive_divergence(self, nepochs = 20, batch_size = 50, cdk = 10, 
									validation = None, auto_save = False, auto_plot = True, decay_ratio = 1.0):
		nbatches = self.input.shape[0] // batch_size
		print('-- Training using contrastive divergence')
		print('nhidden: %d' % self.nhidden)
		print('lrate: %f' % self.lrate)
		print('nepochs: %d' % nepochs)
		print('bsize: %d' % batch_size)
		print('k: %d' % cdk)

		if (validation is not None):
			self.validate_model(validation, 0)

		if(auto_plot):
			im = self.hidden_units_to_image()
			im.save('plots/nhidden_%d_k_%d_cd_filters_epoch_%d.png' % (self.nhidden, cdk, 0))

		for epoch in range(nepochs):
			for batch in range(nbatches):

				x = self.input[batch*batch_size:(batch+1)*batch_size]
				xtilde = self.perform_cd(x, cdk)
				self.update_parameters(x, xtilde)

			'''
				Decreases the learning rate after each epoch
			'''
			self.lrate = self.lrate * decay_ratio

			if(validation is not None):
				self.validate_model(validation, epoch + 1)
		
			if(auto_save):
				self.save_model('k_%d_cd_epoch_%d.model' % (cdk, epoch + 1))	

			if(auto_plot):	
				im = self.hidden_units_to_image()
				im.save('plots/nhidden_%d_k_%d_cd_filters_epoch_%d.png' % (self.nhidden, cdk, epoch + 1))

	def persistent_contrastive_divergence(self, nepochs = 20, batch_size = 50, cdk = 10,
									validation = None, auto_save = False, auto_plot = True, decay_ratio = 1.0):

		nbatches = self.input.shape[0]  // batch_size
		print('-- Training using persistent contrastive divergence')
		print('nhidden: %d' % self.nhidden)
		print('lrate: %f' % self.lrate)
		print('nepochs: %d' % nepochs)
		print('bsize: %d' % batch_size)
		print('k: %d' % cdk)

		xpersisted = self.input.copy()
		
		if (validation is not None):
			self.validate_model(validation, 0) 

		if (auto_plot):
			im = self.hidden_units_to_image()
			im.save('plots/nhidden_%d_k_%d_pcd_filters_epoch_%d.png' % (self.nhidden, cdk, 0))

		for epoch in range(nepochs):
			for batch in range(nbatches):
				low = batch*batch_size
				high = (batch+1)*batch_size
				x = self.input[low:high]

				xpersisted[low:high] = self.perform_cd(xpersisted[low:high], cdk)
							
				self.update_parameters(x, xpersisted[low:high])
			'''
				Decreases the learning rate after each epoch
			'''
			self.lrate = self.lrate * decay_ratio

			if(validation is not None):
				self.validate_model(validation, epoch + 1)

			if(auto_save):
				self.save_model('k_%d_cd_epoch_%d.model' % (cdk, epoch + 1))	
			
			if(auto_plot):	
				im = self.hidden_units_to_image()
				im.save('plots/nhidden_%d_k_%d_pcd_filters_epoch_%d.png' % (self.nhidden, cdk, epoch + 1))

	def validate_model(self, validation, epoch, size = 50):
		print('Epoch %d stats' % (epoch))
		print('\t\tmean Reconstruction error: %f' % self.mean_reconstruction_error(validation))
		print('\t\tmean Cross Entropy: %f' % self.mean_cross_entropy(validation))
		bit_index = np.random.choice(a = validation.shape[1], size = size)
		print('\t\tmean pseudo likelihood: %f' % self.mean_pseudo_likelihood(validation, bit_index))

	def hidden_units_to_image(self, tile_shape = (10, 10)):
		imsize = self.imsize
		tiled_image = tile_raster_images(X = self.W.T, img_shape = (imsize, imsize),
										tile_shape = tile_shape, tile_spacing=(1, 1))
		image = Image.fromarray(tiled_image)
		return image

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
	
	def sample_from_model(self, input = None, samples_to_take = 10, number_of_chains = 10, cdk = 100):
		nsamples = input.shape[0]

		selected_idx = np.random.choice(range(nsamples), samples_to_take)
		digits = input[selected_idx]
		
		tile_shape = (number_of_chains, samples_to_take)
			
		shape = (samples_to_take * number_of_chains, self.imsize * self.imsize) 
		images = np.zeros(shape)
		
		for chain in range(1, number_of_chains):
			digits = self.perform_cd(digits, cdk)
			for idx in range(samples_to_take):
				images[chain*samples_to_take + idx] = digits[idx] 
		


		return Image.fromarray(
										tile_raster_images(images, (self.imsize, self.imsize), (samples_to_take, number_of_chains))
										)
