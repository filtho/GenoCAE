"""GenoCAE.

Usage:
  run_gcae.py train --datadir=<name> --data=<name> --model_id=<name> --train_opts_id=<name> --data_opts_id=<name> --save_interval=<num> --epochs=<num> [--resume_from=<num> --trainedmodeldir=<name> --recomb_rate=<num> --superpops=<name> ] [--pheno_model_id=<name>]
  run_gcae.py project --datadir=<name>   [ --data=<name> --model_id=<name>  --train_opts_id=<name> --data_opts_id=<name> --superpops=<name> --epoch=<num> --trainedmodeldir=<name>   --pdata=<name> --trainedmodelname=<name> --alt_data=<name>]
  run_gcae.py plot --datadir=<name> [  --data=<name>  --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name>  --pdata=<name> --trainedmodelname=<name>]
  run_gcae.py animate --datadir=<name>   [ --data=<name>   --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name> --pdata=<name> --trainedmodelname=<name>]
  run_gcae.py evaluate --datadir=<name> --metrics=<name>  [  --data=<name>  --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name>  --pdata=<name> --trainedmodelname=<name>]

Options:
  -h --help             show this screen
  --datadir=<name>      directory where sample data is stored. if not absolute: assumed relative to GenoCAE/ directory.
  --data=<name>         file prefix, not including path, of the data files (EIGENSTRAT of PLINK format)
  --trainedmodeldir=<name>     base path where to save model training directories. if not absolute: assumed relative to GenoCAE/ directory. default: ae_out/
  --model_id=<name>     model id, corresponding to a file models/model_id.json
  --train_opts_id=<name>train options id, corresponding to a file train_opts/train_opts_id.json
  --data_opts_id=<name> data options id, corresponding to a file data_opts/data_opts_id.json
  --epochs<num>         number of epochs to train
  --resume_from<num>	saved epoch to resume training from. set to -1 for latest saved epoch.
  --save_interval<num>	epoch intervals at which to save state of model
  --trainedmodelname=<name> name of the model training directory to fetch saved model state from when project/plot/evaluating
  --pdata=<name>     	file prefix, not including path, of data to project/plot/evaluate. if not specified, assumed to be the same the model was trained on.
  --epoch<num>          epoch at which to project/plot/evaluate data. if not specified, all saved epochs will be used
  --superpops<name>     path+filename of file mapping populations to superpopulations. used to color populations of the same superpopulation in similar colors in plotting. if not absolute path: assumed relative to GenoCAE/ directory.
  --metrics=<name>      the metric(s) to evaluate, e.g. hull_error of f1 score. can pass a list with multiple metrics, e.g. "hull_error,f1_score"

  --alt_data=<name> 	project a model on another dataset than it was trained on

"""

from docopt import docopt, DocoptExit
import tensorflow as tf
from tensorflow.keras import Model, layers
from datetime import datetime
from utils.data_handler_recombined import  get_saved_epochs, get_projected_epochs, write_h5, read_h5, get_coords_by_pop, data_generator_ae, convex_hull_error, f1_score_kNN, plot_genotype_hist, to_genotypes_sigmoid_round, to_genotypes_invscale_round, GenotypeConcordance, get_pops_with_k, get_ind_pop_list_from_map, get_baseline_gc, write_metric_per_epoch_to_csv
from utils.visualization import plot_coords_by_superpop, plot_clusters_by_superpop, plot_coords, plot_coords_by_pop, make_animation, write_f1_scores_to_csv
import utils.visualization
import utils.layers
import json
import numpy as np
import pandas as pd
import time
import os
import math
import matplotlib.pyplot as plt
import csv
import copy
import h5py
import matplotlib.animation as animation
from pathlib import Path
from utils.data_handler_recombined import alt_data_generator, parquet_converter
from utils.set_tf_config_berzelius import set_tf_config


#tf.config.experimental.enable_tensor_float_32_execution(False)
k_vec = [1,2,3,4,5]
def chief_print(str):

    if "isChief" in os.environ:

        if os.environ["isChief"] == "true":
            print(str)
    else:
        print(str)

def _isChief():

	if "isChief" in os.environ:

		if os.environ["isChief"] == "true":
			return True
		else:
			return False
	else:
		return True


GCAE_DIR = Path(__file__).resolve().parent
class Autoencoder(Model):


	def __init__(self, model_architecture, n_markers, noise_std, regularizer):
		'''

		Initiate the autoencoder with the specified options.
		All variables of the model are defined here.

		:param model_architecture: dict containing a list of layer representations
		:param n_markers: number of markers / SNPs in the data
		:param noise_std: standard deviation of noise to add to encoding layer during training. False if no noise.
		:param regularizer: dict containing regularizer info. False if no regularizer.
		'''
		super(Autoencoder, self).__init__()
		self.all_layers = []
		self.n_markers = n_markers
		self.noise_std = noise_std
		self.residuals = dict()
		self.marker_spec_var = False
		self.passthrough = tf.Variable(1.0, dtype=tf.float32, trainable=False)

		self.ge = tf.random.Generator.from_seed(1)

		chief_print("\n______________________________ Building model ______________________________")
		# variable that keeps track of the size of layers in encoder, to be used when constructing decoder.
		ns=[]
		ns.append(n_markers)

		first_layer_def = model_architecture["layers"][0]
		layer_module = getattr(eval(first_layer_def["module"]), first_layer_def["class"])
		layer_args = first_layer_def["args"]
		for arg in ["n", "size", "layers", "units", "shape", "target_shape", "output_shape", "kernel_size", "strides"]:

				if arg in layer_args.keys():
					layer_args[arg] = eval(str(layer_args[arg]))

		if "kernel_initializer" in layer_args and layer_args["kernel_initializer"] == "flum":
			if "kernel_size" in layer_args:
				dim = layer_args["kernel_size"] * layer_args["filters"]
			else:
				dim = layer_args["units"]
			limit = math.sqrt(2 * 3 / (dim))
			layer_args["kernel_initializer"] = tf.keras.initializers.RandomUniform(-limit, limit)

		if "kernel_regularizer" in layer_args and layer_args["kernel_regularizer"] == "L2":

				layer_args["kernel_regularizer"] = tf.keras.regularizers.L2(l2=0.0)

		try:
			activation = getattr(tf.nn, layer_args["activation"])
			layer_args.pop("activation")
			first_layer = layer_module(activation=activation, **layer_args)

		except KeyError:
			first_layer = layer_module(**layer_args)
			activation = None

		self.all_layers.append(first_layer)
		chief_print("Adding layer: " + str(layer_module.__name__) + ": " + str(layer_args))

		if first_layer_def["class"] == "conv1d" and "strides" in layer_args.keys() and layer_args["strides"] > 1:
			ns.append(int(first_layer.shape[1]))
			raise NotImplementedError

		# add all layers except first
		for layer_def in model_architecture["layers"][1:]:
			layer_module = getattr(eval(layer_def["module"]), layer_def["class"])
			layer_args = layer_def["args"]

			for arg in ["n", "size", "layers", "units", "shape", "target_shape", "output_shape", "kernel_size", "strides"]:

				if arg in layer_args.keys():
					layer_args[arg] = eval(str(layer_args[arg]))

			if "kernel_initializer" in layer_args and layer_args["kernel_initializer"] == "flum":
				if "kernel_size" in layer_args:
					dim = layer_args["kernel_size"] * layer_args["filters"]
				else:
					dim = layer_args["units"]
				limit = math.sqrt(2 * 3 / (dim))
				layer_args["kernel_initializer"] = tf.keras.initializers.RandomUniform(-limit, limit)

			if "kernel_regularizer" in layer_args and layer_args["kernel_regularizer"] == "L2":
				layer_args["kernel_regularizer"] = tf.keras.regularizers.L2(l2=1e-6)

			if layer_def["class"] == "MaxPool1D":
				ns.append(int(math.ceil(float(ns[-1]) / layer_args["strides"])))

			if layer_def["class"] == "Conv1D" and "strides" in layer_def.keys() and layer_def["strides"] > 1:
				raise NotImplementedError

			chief_print("Adding layer: " + str(layer_module.__name__) + ": " + str(layer_args))

			if "name" in layer_args and (layer_args["name"] == "i_msvar" or layer_args["name"] == "nms"):
				self.marker_spec_var = True



			if "activation" in layer_args.keys():
				activation = getattr(tf.nn, layer_args["activation"])
				layer_args.pop("activation")
				this_layer = layer_module(activation=activation, **layer_args)
			else:
				this_layer = layer_module(**layer_args)


			self.all_layers.append(this_layer)

		if noise_std:
			self.noise_layer = tf.keras.layers.GaussianNoise(noise_std)

		self.ns = ns
		self.regularizer = regularizer

		if self.marker_spec_var:
			random_uniform = tf.random_uniform_initializer()
			self.ms_variable = tf.Variable(random_uniform(shape = (1, n_markers), dtype=tf.float32))#, name="marker_spec_var")
			self.nms_variable = tf.Variable(random_uniform(shape = (1, n_markers), dtype=tf.float32))#, name="nmarker_spec_var")
		else:
			chief_print("No marker specific variable.")

	def call(self, input_data, targets=None, is_training = True, verbose = False, rander=[False, False], regloss=True):
		'''
		The forward pass of the model. Given inputs, calculate the output of the model.

		:param input_data: input data
		:param is_training: if called during training
		:param verbose: print the layers and their shapes
		:return: output of the model (last layer) and latent representation (encoding layer)

		'''

		# if we're adding a marker specific variables as an additional channel
		if self.marker_spec_var:
			# Tiling it to handle the batch dimension

			ms_tiled = tf.tile(self.ms_variable, (tf.shape(input_data)[0], 1))
			ms_tiled = tf.expand_dims(ms_tiled, 2)
			nms_tiled = tf.tile(self.nms_variable, (tf.shape(input_data)[0], 1))
			nms_tiled = tf.expand_dims(nms_tiled, 2)
			concatted_input = tf.concat([input_data, ms_tiled], 2)
			input_data = concatted_input

		if verbose:
			chief_print("inputs shape " + str(input_data.shape))

		first_layer = self.all_layers[0]
		counter = 1

		if verbose:
			chief_print("layer {0}".format(counter))
			chief_print("--- type: {0}".format(type(first_layer)))

		x = first_layer(inputs=input_data)

		if "Residual" in first_layer.name:
			out = self.handle_residual_layer(first_layer.name, x, verbose=verbose)
			if not out == None:
				x = out
		if verbose:
			chief_print("--- shape: {0}".format(x.shape))

		# indicator if were doing genetic clustering (ADMIXTURE-style) or not
		have_encoded_raw = False
		encoded_data = None

		# do all layers except first
		for layer_def in self.all_layers[1:]:
			try:
				layer_name = layer_def.cname
			except:
				layer_name = layer_def.name

			counter += 1

			if verbose:
				chief_print("layer {0}: {1} ({2}) ".format(counter, layer_name, type(layer_def)))

			if layer_name == "dropout":
				x = layer_def(x, training = is_training)
			else:
				x = layer_def(x, training = is_training)

			# If this is a clustering model then we add noise to the layer first in this step
			# and the next layer, which is sigmoid, is the actual encoding.
			if layer_name == "encoded_raw":
				have_encoded_raw = True
				if self.noise_std:
					x = self.noise_layer(x, training = is_training)
				encoded_data_raw = x

			# If this is the encoding layer, we add noise if we are training
			elif "encoded" in layer_name:
				if self.noise_std and not have_encoded_raw:
					x = self.noise_layer(x, training = is_training)
				###x += 50.
				encoded_data_pure = x

				#rand_data = self.ge.uniform(tf.shape(x), minval=0., maxval=100.0)
				#TODO The call x[:,i] below in "rander" may result in slicing arrays in a dimension with 0 elements, which can happen in multi gpu training
				# when the number of gpus is larger than the size of the last batch. This means that the some GPUs may have empty batches. Unsure how to get a nice solution

				#rander = rander + [False] * (len(x[0,:]) - len(rander))
				#x = tf.stack([rand_data[:,i] if dorand else x[:,i] for i, dorand in enumerate(rander)], axis=1)
				##x = tf.math.mod(x, 100.)
				encoded_data = x


				flipsquare = False
				if self.regularizer and "flipsquare" in self.regularizer:
					flipsquare = self.regularizer["flipsquare"]
				##x = x + tf.where(x < ge.uniform(tf.shape(x), minval=0, maxval=100.0), 100.0, 0.)

				#x = tf.concat((x, y), axis=-1)
				#, (x*x) * (tf.sign(x) if flipsquare else 1.0), (y*y) * (tf.sign(y) if flipsquare else 1.0)
				if "basis" in layer_name:
					basis = tf.expand_dims(tf.range(-200., 200., 20.), axis=0)
					x = tf.clip_by_value(tf.concat([tf.expand_dims(x[:,i], axis=1) - basis for i in range(2)], axis=1), -40., 40.)
				#x = tf.where(self.passthrough == 1.0, x, tf.stop_gradient(x))



			if "Residual" in layer_name:
				out = self.handle_residual_layer(layer_name, x, verbose=verbose)
				if not out == None:
					x = out

			# inject marker-specific variable by concatenation
			if "i_msvar" in layer_name and self.marker_spec_var:
				x = self.injectms(verbose, x, layer_name, ms_tiled, self.ms_variable)

			if "nms" in layer_name and self.marker_spec_var:
				x = self.injectms(verbose, x, layer_name, nms_tiled, self.nms_variable)

			if verbose:
				chief_print("--- shape: {0}".format(x.shape))

		if self.regularizer and regloss and encoded_data is not None:
			reg_module = eval(self.regularizer["module"])
			reg_name = getattr(reg_module, self.regularizer["class"])
			reg_func = reg_name(float(self.regularizer["reg_factor"]))

			# if this is a clustering self then the regularization is added to the raw encoding, not the softmaxed one
			#if have_encoded_raw:
			#	reg_loss = reg_func(encoded_data_raw)
			#else:
			#	reg_loss = reg_func(encoded_data)


			#TODO:  I have this idea that we should give each samples 1 unit square of of space in the 2D representation
			# Thought, how to translagte this when using recombined samples...?


			reg_loss = self.regularizer["reg_factor"] * tf.reduce_sum(tf.math.maximum(0., tf.square(encoded_data_pure) - 1 * 40000.))
			reg_loss = self.regularizer["reg_factor"] * tf.reduce_sum(tf.math.maximum(0., tf.square(encoded_data_pure) - 1 * 2000))

			#reg_loss += self.regularizer["reg_factor"] * tf.reduce_sum(tf.math.maximum(0., tf.square(x) - 1 * 36.))
			self.add_loss(reg_loss)
		if targets is not None and False:
			reploss = tf.constant(0., tf.float32)
			for i in range(1, tf.shape(encoded_data)[0] - 1):
				shifted = tf.stop_gradient(tf.roll(encoded_data, i, axis=0))
				shifted2 = tf.stop_gradient(tf.roll(encoded_data, i + 1, axis=0))
				shifted_targets = tf.stop_gradient(tf.roll(targets, i, axis=0))
				diff = encoded_data - shifted
				##diff = tf.math.mod(diff, 100.)
				##diff += tf.where(diff < -50., 100., 0.)
				##diff += tf.where(diff > 50., -100., 0.)
				#diff = tf.where(tf.expand_dims(tf.sign(diff[:,0]) >= 0, axis = -1), diff, 0.)
				mean = tf.math.reduce_mean(encoded_data, axis=0, keepdims=True)
				#diff *= tf.expand_dims(tf.where(tf.norm(shifted - mean, axis = -1) < tf.norm(encoded_data - mean, axis = -1), 1.0, 0.0), axis=-1)
				smalleralong = tf.math.reduce_sum(tf.square(encoded_data - mean), axis = -1) < tf.math.reduce_sum((encoded_data - mean) * (shifted - mean), axis = -1)
				mismatch = tf.math.reduce_mean(tf.where(targets == shifted_targets, 0.0, 1.0), axis=-1)
				#diff *= tf.expand_dims(tf.where(smalleralong, 0.0, 1.0), axis=-1)
				#norm = tf.expand_dims(tf.norm(diff, ord = 2, axis = -1), axis=-1)
				# tf.stop_gradient(diff / (norm + 1e-19)) *
				r2 = (tf.norm(diff, ord = self.regularizer["ord"], axis = -1))**tf.cast(self.regularizer["ord"], tf.float32) + self.regularizer["max_rep"]
				#r2 *= 0.0001
				##reploss += tf.math.reduce_sum(self.regularizer["rep_factor"] * (mismatch * tf.math.exp(-r2 * 0.2)) - 0.02 * tf.math.exp(-r2*0.5*0.2) - 0.02 * tf.math.exp(-r2*0.05*0.2))
				#reploss += tf.math.reduce_sum(self.regularizer["rep_factor"] * tf.math.maximum(0., 0.5 - mismatch * -r2))
				reploss += tf.math.reduce_sum(self.regularizer["rep_factor"] * tf.math.maximum(0., 30. * mismatch - r2))
				shiftedc = (shifted + shifted2)*0.5
				##shiftedc = (tf.math.mod(shifted, 100.) + tf.math.mod(shifted2, 100.)) * 0.5

				shifteddiff = (shifted - shifted2)
				##shifteddiff = tf.math.mod(shifteddiff, 100.)
				##shifteddiff += tf.where(shifteddiff < -50., 100., 0.)
				##shifteddiff += tf.where(shifteddiff > 50., -100., 0.)
				if False:
					shifteddiff = tf.stack((-shifteddiff[:,1], shifteddiff[:,0]), axis=1)
					seconddiff = encoded_data - shiftedc
					seconddiff = tf.math.mod(diff, 100.)
					seconddiff += tf.where(diff < -50., 100., 0.)
					seconddiff += tf.where(diff > 50., -100., 0.)
					seconddiff *= shifteddiff
					seconddiff /= tf.norm(shifteddiff) + 1e-9
				else:
					seconddiff = encoded_data - shiftedc
					seconddiff -= shifteddiff * tf.math.reduce_sum(seconddiff * shifteddiff, axis=-1, keepdims=True) / (tf.norm(shifteddiff) + 1e-9)**2
				diff = seconddiff
				r2 = (tf.norm(diff, ord = self.regularizer["ord"], axis = -1))**tf.cast(self.regularizer["ord"], tf.float32) + self.regularizer["max_rep"]
				#r2 *= 0.0001
				#reploss += tf.math.reduce_sum(self.regularizer["rep_factor"] * tf.math.exp(-r2 * 0.2))


				#self.add_loss(tf.math.reduce_sum(self.regularizer["rep_factor"] * (tf.math.reduce_mean(tf.where(targets == shifted_targets, 0.0, 1.0), axis=-1) * r2**-6.0 - r2**-3.0)))
				# tf.norm(diff, ord = 2, axis = -1)
				# * f.math.l2_normalize(diff, axis = -1)
			tf.print(reploss)
			self.add_loss(reploss)
		return x, encoded_data


	def handle_residual_layer(self, layer_name, input, verbose=False):
		suffix = layer_name.split("Residual_")[-1].split("_")[0]
		res_number = suffix[0:-1]
		if suffix.endswith("a"):
			if verbose:
				chief_print("encoder-to-decoder residual: saving residual {}".format(res_number))
			self.residuals[res_number] = input
			return None
		if suffix.endswith("b"):
			if verbose:
				chief_print("encoder-to-decoder residual: adding residual {}".format(res_number))
			residual_tensor = self.residuals[res_number]
			res_length = residual_tensor.shape[1]
			if len(residual_tensor.shape) == 3:
				x = tf.keras.layers.Add()([input[:,0:res_length,:], residual_tensor])
			if len(residual_tensor.shape) == 2:
				x = tf.keras.layers.Add()([input[:,0:res_length], residual_tensor])

			return x

	def injectms(self, verbose, x, layer_name, ms_tiled, ms_variable):
		if verbose:
				chief_print("----- injecting marker-specific variable")

		# if we need to reshape ms_variable before concatting it
		if not self.n_markers == x.shape[1]:
				d = int(math.ceil(float(self.n_markers) / int(x.shape[1])))
				diff = d*int(x.shape[1]) - self.n_markers
				ms_var = tf.reshape(tf.pad(ms_variable,[[0,0],[0,diff]]), (-1, x.shape[1],d))
				# Tiling it to handle the batch dimension
				ms_tiled = tf.tile(ms_var, (tf.shape(x)[0],1,1))

		else:
				# Tiling it to handle the batch dimension
				ms_tiled = tf.tile(ms_variable, (x.shape[0],1))
				ms_tiled = tf.expand_dims(ms_tiled, 2)

		if "_sg" in layer_name:
				if verbose:
						chief_print("----- stopping gradient for marker-specific variable")
				ms_tiled = tf.stop_gradient(ms_tiled)


		if verbose:
				chief_print("ms var {}".format(ms_variable.shape))
				chief_print("ms tiled {}".format(ms_tiled.shape))
				chief_print("concatting: {0} {1}".format(x.shape, ms_tiled.shape))

		x = tf.concat([x, ms_tiled], 2)


		return x


def get_batches(n_samples, batch_size):
	n_batches = n_samples // batch_size

	n_samples_last_batch = n_samples % batch_size
	if n_samples_last_batch > 0:
		n_batches += 1
	else:
		n_samples_last_batch = batch_size

	return n_batches, n_samples_last_batch

def alfreqvector(y_pred):
	'''
	Get a probability distribution over genotypes from y_pred.
	Assumes y_pred is raw model output, one scalar value per genotype.

	Scales this to (0,1) and interprets this as a allele frequency, uses formula
	for Hardy-Weinberg equilibrium to get probabilities for genotypes [0,1,2].

	TODO: Currently not current, using logits values in some cases.

	:param y_pred: (n_samples x n_markers) tensor of raw network output for each sample and site
	:return: (n_samples x n_markers x 3 tensor) of genotype probabilities for each sample and site
	'''

	if len(y_pred.shape) == 2:
		alfreq = tf.keras.activations.sigmoid(y_pred)
		alfreq = tf.expand_dims(alfreq, -1)
		return tf.concat(((1-alfreq) ** 2, 2 * alfreq * (1 - alfreq), alfreq ** 2), axis=-1)
		#return tf.concat(((1-alfreq), alfreq), axis=-1)
	else:
		return y_pred[:,:,0:3]#tf.nn.softmax(y_pred)




def generatepheno(data, poplist):
		if data is None:
			return None
		return tf.expand_dims(tf.convert_to_tensor([data.get((fam, name), None) for name, fam in poplist]), axis=-1)

def readpheno(file, num):
	with open(file, "rt") as f:
		for _ in f:
			break
		return {(line[0], line[1]) : float(line[num + 2]) for line in (full_line.split() for full_line in f)}

def writephenos(file, poplist, phenos):
	if _isChief():
		with open(file, "wt") as f:
			for (name, fam), pheno in zip(poplist, phenos):
				f.write(f'{fam} {name} {pheno}\n')


def save_weights(train_directory, prefix, model):
	if model is None:
		return

	if _isChief():
		if os.path.isdir(prefix):
			newname = train_directory+"_"+str(time.time())
			os.rename(train_directory, newname)
			print("... renamed " + train_directory + " to  " + newname)

		model.save_weights(prefix, save_format ="tf")



def main():
	chief_print("tensorflow version {0}".format(tf.__version__))
	tf.keras.backend.set_floatx('float32')

	try:
		arguments = docopt(__doc__, version='GenoAE 1.0')
	except DocoptExit:
		chief_print("Invalid command. Run 'python run_gcae.py --help' for more information.")
		exit(1)

	for k in list(arguments.keys()):
		knew = k.split('--')[-1]
		arg=arguments.pop(k)
		arguments[knew]=arg

	if "SLURMD_NODENAME" in os.environ:

		slurm_job = 1
		addresses, chief, num_workers = set_tf_config()
		isChief = os.environ["SLURMD_NODENAME"] == chief
		os.environ["isChief"] = json.dumps(str(isChief))
		chief_print(num_workers)
		if num_workers > 1 and arguments["train"]:
			#Here, NCCL is what I would want to use - it is Nvidias own implementation of reducing over the devices. However, this induces a segmentation fault, and the default value works.
			# had some issues, I think this was resolved by updating to TensorFlow 2.7 from 2.5.
			# However, the image is built on the cuda environment for 2.5 This leads to problems when profiling


			strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver(),
																communication_options=tf.distribute.experimental.CommunicationOptions(
																implementation=tf.distribute.experimental.CollectiveCommunication.NCCL)
																)
			tf.print(tf.config.list_physical_devices(device_type='GPU'))
			num_physical_gpus = len(tf.config.list_physical_devices(device_type='GPU'))

			chief_print(tf.config.list_physical_devices(device_type='GPU'))
			gpus = ["gpu:"+ str(i) for i in range(num_physical_gpus)]
			chief_print(gpus)


		else:
			if not isChief:
				print("Work has ended for this worker, now relying only on the Chief :)")
				exit(0)
			tf.print(tf.config.list_physical_devices(device_type='GPU'))
			tf.print(tf.test.gpu_device_name())
			num_physical_gpus = len(tf.config.list_physical_devices(device_type='GPU'))

			chief_print(tf.config.list_physical_devices(device_type='GPU'))
			gpus = ["gpu:"+ str(i) for i in range(num_physical_gpus)]
			chief_print(gpus)
			strategy =  tf.distribute.MirroredStrategy(devices = gpus, cross_device_ops=tf.distribute.NcclAllReduce())

			slurm_job = 0
		os.environ["isChief"] = json.dumps((isChief))

	else:
		isChief = True
		slurm_job = 0
		num_workers = 1

		strategy =  tf.distribute.MirroredStrategy()


	@tf.function
	def project_batch(autoencoder, loss_func, input_train_batch, targets_train_batch, pheno_model):
		decoded_train_batch, encoded_train_batch = autoencoder(input_train_batch, is_training = True)
		if pheno_model is not None:
			add = pheno_model(encoded_train_batch)[0][:,0]
			print(add)
			print(np.shape(add))
			if pheno_train is not None:
				pheno_train = np.concatenate((pheno_train, add), axis=0)
			else:
				pheno_train = add

		loss_train_batch = loss_func(y_pred = decoded_train_batch, y_true = targets_train_batch)
		#loss_train_batch += sum(autoencoder.losses)
		return decoded_train_batch, encoded_train_batch, loss_train_batch


	@tf.function
	def init_model(autoencoder, input_test):
		return autoencoder(input_test[0:2], is_training = False, verbose = True)

	chief_print("slurm_job" + str(slurm_job))
	print(isChief)

	num_devices = strategy.num_replicas_in_sync
	chief_print('Number of devices: {}'.format(num_devices))



	if arguments["trainedmodeldir"]:
		trainedmodeldir = arguments["trainedmodeldir"]
		if not os.path.isabs(trainedmodeldir):
			trainedmodeldir="{}/{}/".format(GCAE_DIR, trainedmodeldir)

	else:
		trainedmodeldir="{}/ae_out/".format(GCAE_DIR)

	if arguments["datadir"]:
		datadir = arguments["datadir"]
		if not os.path.isabs(datadir):
			datadir="{}/{}/".format(GCAE_DIR, datadir)

	else:
		datadir="{}/data/".format(GCAE_DIR)

	if arguments["trainedmodelname"]:
		trainedmodelname = arguments["trainedmodelname"]
		train_directory = trainedmodeldir + trainedmodelname

		split = trainedmodelname.split(".")

		data_opts_id = split[3]
		train_opts_id = split[2]
		model_id = split[1]
		data = split[4]
		pheno_model_id = (split + [None])[5]
	else:
		data = arguments['data']
		data_opts_id = arguments["data_opts_id"]
		train_opts_id = arguments["train_opts_id"]
		model_id = arguments["model_id"]
		pheno_model_id = arguments.get("pheno_model_id")

		train_directory = False

	if arguments["recomb_rate"]:
		recomb_rate = float(arguments["recomb_rate"])
	else:
		recomb_rate = 0

	if arguments["alt_data"]:
		alt_data = arguments["alt_data"]
	else:
		alt_data = None


	with open("{}/data_opts/{}.json".format(GCAE_DIR, data_opts_id)) as data_opts_def_file:
		data_opts = json.load(data_opts_def_file)

	with open("{}/train_opts/{}.json".format(GCAE_DIR, train_opts_id)) as train_opts_def_file:
		train_opts = json.load(train_opts_def_file)

	with open("{}/models/{}.json".format(GCAE_DIR, model_id)) as model_def_file:
		model_architecture = json.load(model_def_file)

	if pheno_model_id is not None:
		with open(f"{GCAE_DIR}/models/{pheno_model_id}.json") as model_def_file:
			pheno_model_architecture = json.load(model_def_file)
	else:
		pheno_model_architecture = None

	for layer_def in model_architecture["layers"]:
		if "args" in layer_def.keys() and "name" in layer_def["args"].keys() and "encoded" in layer_def["args"]["name"] and "units" in layer_def["args"].keys():
			n_latent_dim = layer_def["args"]["units"]

	# indicator of whether this is a genetic clustering or dimensionality reduction model
	doing_clustering = False
	for layer_def in model_architecture["layers"][1:-1]:
		if "encoding_raw" in layer_def.keys():
			doing_clustering = True

	chief_print("\n______________________________ arguments ______________________________")
	for k in arguments.keys():
		chief_print(k + " : " + str(arguments[k]))
	chief_print("\n______________________________ data opts ______________________________")
	for k in data_opts.keys():
		chief_print(k + " : " + str(data_opts[k]))
	chief_print("\n______________________________ train opts ______________________________")
	for k in train_opts:
		chief_print(k + " : " + str(train_opts[k]))
	chief_print("______________________________")


	batch_size = train_opts["batch_size"] # * num_devices
	learning_rate = train_opts["learning_rate"]
	regularizer = train_opts["regularizer"]

	superpopulations_file = arguments['superpops']
	if superpopulations_file and not os.path.isabs(os.path.dirname(superpopulations_file)):
		superpopulations_file="{}/{}/{}".format(GCAE_DIR, os.path.dirname(superpopulations_file), Path(superpopulations_file).name)

	norm_opts = data_opts["norm_opts"]
	norm_mode = data_opts["norm_mode"]
	validation_split = data_opts["validation_split"]

	try:
		holdout_val_pop = data_opts["holdout_val_pop"]
	except:

		holdout_val_pop = None


	if "sparsifies" in data_opts.keys():
		sparsify_input = True
		missing_mask_input = True
		n_input_channels = 2
		sparsifies = data_opts["sparsifies"]

	else:
		sparsify_input = False
		missing_mask_input = False
		n_input_channels = 1

	if "impute_missing" in data_opts.keys():
		fill_missing = data_opts["impute_missing"]

	else:
		fill_missing = False

	if fill_missing:
		chief_print("Imputing originally missing genotypes to most common value.")
	else:
		chief_print("Keeping originally missing genotypes.")
		missing_mask_input = True
		n_input_channels = 2

	if not train_directory:
		dirparts = [model_id, train_opts_id, data_opts_id, data] + ([pheno_model_id] if pheno_model_id is not None else [])
		train_directory = trainedmodeldir + "ae." + ".".join(dirparts)

	if arguments["pdata"]:
		pdata = arguments["pdata"]
	else:
		pdata = data

	data_prefix = datadir + pdata
	results_directory = "{0}/{1}".format(train_directory, pdata)
	if alt_data is not None:
		results_directory = train_directory+"/"+str(alt_data)
	try:
		os.mkdir(results_directory)
	except OSError:
		pass

	encoded_data_file = "{0}/{1}/{2}".format(train_directory, pdata, "encoded_data.h5")

	if "noise_std" in train_opts.keys():
		noise_std = train_opts["noise_std"]
	else:
		noise_std = False

	if (arguments['evaluate'] or arguments['animate'] or arguments['plot']):

		if os.path.isfile(encoded_data_file):
			encoded_data = h5py.File(encoded_data_file, 'r')
		else:
			print("------------------------------------------------------------------------")
			print("Error: File {0} not found.".format(encoded_data_file))
			print("------------------------------------------------------------------------")
			exit(1)

		epochs = get_projected_epochs(encoded_data_file)

		if arguments['epoch']:
			epoch = int(arguments['epoch'])
			if epoch in epochs:
				epochs = [epoch]
			else:
				print("------------------------------------------------------------------------")
				print("Error: Epoch {0} not found in {1}.".format(epoch, encoded_data_file))
				print("------------------------------------------------------------------------")
				exit(1)

		if doing_clustering:
			if arguments['animate']:
				print("------------------------------------------------------------------------")
				print("Error: Animate not supported for genetic clustering model.")
				print("------------------------------------------------------------------------")
				exit(1)


			if arguments['plot'] and not superpopulations_file:
				print("------------------------------------------------------------------------")
				print("Error: Plotting of genetic clustering results requires a superpopulations file.")
				print("------------------------------------------------------------------------")
				exit(1)

	else:
		max_mem_size = 5 * 10**10 # this is approx 50GB
		if alt_data is not None:
			data_prefix = datadir + alt_data

		filebase = data_prefix
		parquet_converter(filebase, max_mem_size=max_mem_size)
		data = alt_data_generator(filebase= data_prefix,
						batch_size = batch_size,
						normalization_mode = norm_mode,
						normalization_options = norm_opts,
						impute_missing = fill_missing)

		n_markers = data.n_markers
		if pheno_model_architecture is not None:
			phenodata = readpheno(data_prefix + ".phe", 2)
		else:
			phenodata = None


		loss_def = train_opts["loss"]
		loss_class = getattr(eval(loss_def["module"]), loss_def["class"])
		if "args" in loss_def.keys():
			loss_args = loss_def["args"]
		else:
			loss_args = dict()
		loss_obj = loss_class(**loss_args)

		def get_originally_nonmissing_mask(genos):
			'''
			Get a boolean mask representing missing values in the data.
			Missing value is represented by float(norm_opts["missing_val"]).

			Uses the presence of missing_val in the true genotypes as indicator, missing_val should not be set to
			something that can exist in the data set after normalization!!!!

			:param genos: (n_samples x n_markers) genotypes
			:return: boolean mask of the same shape as genos
			'''
			orig_nonmissing_mask = tf.not_equal(genos, float(norm_opts["missing_val"]))

			return orig_nonmissing_mask

		with strategy.scope():
			if loss_class == tf.keras.losses.CategoricalCrossentropy or loss_class == tf.keras.losses.KLDivergence or True:



				def loss_func(y_pred, y_true, pow=1., avg=False):
					y_pred = y_pred[:, 0:n_markers]
					#y_pred = alfreqvector(y_pred)
					#y_true = tf.one_hot(tf.cast(y_true * 2, tf.uint8), 3)
					#y1 = y_true[:, :, 0] + 0.5 * y_true[:, :, 1]
					#y_true = tf.stack((y1, 1.0 - y1), axis = -1)



					#if not fill_missing:
					#	orig_nonmissing_mask = get_originally_nonmissing_mask(y_true)
					#else:
					#	orig_nonmissing_mask = np.full(y_true.shape, True)
					# TODO: Reintroduce missingness support here, with proper shape after slicing!

					y_trueorig = y_true
					#tf.print(tf.shape(y_true))
					y_pred = alfreqvector(y_pred)
					y_true = tf.one_hot(tf.cast(y_true * 2, tf.uint8), 3) #* 0.9997 + 0.0001

					#tf.print(y_trueorig)
					#tf.print(tf.reduce_max(y_trueorig))
					#tf.print(tf.reduce_min(y_trueorig))

					y_true2 = y_true#*0.997 + 0.001
					if avg:
						y_true = y_true * -1./tf.cast(tf.shape(y_true)[0], tf.float32) + tf.math.reduce_mean(y_true, axis=0, keepdims=True)

					#tf.print("YPRED", y_pred)
					#tf.print("YTRUE", y_true)
					#tf.print("YMASK", orig_nonmissing_mask)
					#tf.print("YMASK", tf.math.zero_fraction(orig_nonmissing_mask))
					#tf.print("PRED", y_pred[orig_nonmissing_mask,:])
					#tf.print("TRUE", y_true[orig_nonmissing_mask,:])
					#return -tf.math.reduce_mean(tf.math.reduce_sum(tf.math.log(y_pred+1e-30) * y_true, axis = 0) / ((tf.math.reduce_sum(y_true2, axis=0))))
					beta = 0.999

					###return -tf.math.reduce_mean(tf.math.reduce_sum(tf.math.log(y_pred+1e-30) * y_true, axis = 0) * (1.0 - beta) / (1-tf.math.pow(beta, tf.math.reduce_sum(y_true2, axis=0)+1)+1e-9))
					#return -tf.math.reduce_mean(tf.math.reduce_sum(      (tf.clip_by_value(y_pred,-10,10)-tf.math.reduce_max(y_pred, axis=-1, keepdims=True)) * y_true, axis = 0) * (1.0 - beta) / (1-tf.math.pow(beta, tf.math.reduce_sum(y_true2, axis=0)+1)+1e-9))
					y_pred *= pow
					y_pred = y_pred - tf.stop_gradient(tf.math.reduce_max(y_pred, axis=-1, keepdims=True))
					y_pred_prob = tf.nn.softmax(y_pred)



					gamma = 1.0
					#partialres = (tf.math.reduce_sum(      (y_pred-tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), axis=-1, keepdims=True))) * y_true * (1 - tf.math.reduce_sum(tf.stop_gradient(y_pred_prob) * y_true, axis=-1, keepdims=True)/tf.math.reduce_sum(y_true * y_true, axis=-1, keepdims=True))**gamma, axis = 0) * (1.0 - beta) / (1-tf.math.pow(beta, tf.math.reduce_sum(y_true2, axis=0)+1)+1e-9))
					partialres = - (tf.math.reduce_sum(      (y_pred-tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), axis=-1, keepdims=True))) * y_true * (1 - tf.math.reduce_sum(tf.stop_gradient(y_pred_prob) * y_true, axis=-1, keepdims=True)/tf.math.reduce_sum(y_true * y_true, axis=-1, keepdims=True))**gamma, axis = 0) * tf.math.pow(beta, tf.math.reduce_sum(y_true2, axis=0)-1))
					##partialres = (tf.math.reduce_sum(      (y_pred-tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), axis=-1, keepdims=True))) * y_true, axis = 0) * (1.0 - beta) / (1-tf.math.pow(beta, tf.math.reduce_sum(y_true2, axis=0)+1)+1e-9))

					##return -tf.math.reduce_mean(tf.boolean_mask(partialres, ge.uniform(tf.shape(partialres)) < 0.9))
					return tf.math.reduce_mean(partialres)
					#return tf.nn.compute_average_loss(partialres, global_batch_size=batch_size)
					#return 0.5 * loss_obj(y_pred = y_pred, y_true = y_true) + 0.5 * loss_obj(y_pred = tf.math.reduce_mean(y_pred, axis = 0, keepdims = True), y_true = tf.math.reduce_mean(y_true, axis = 0, keepdims = True))



			else:
				def loss_func(y_pred, y_true):

					y_pred = y_pred[:, 0:n_markers]
					y_true = tf.convert_to_tensor(y_true)

					if not fill_missing:
						orig_nonmissing_mask = get_originally_nonmissing_mask(y_true)
						y_pred = y_pred[orig_nonmissing_mask]
						y_true = y_true[orig_nonmissing_mask]

					return loss_obj(y_pred = y_pred, y_true = y_true)


			@tf.function
			def distributed_train_step(model, model2, optimizer, optimizer2, loss_function, input, targets, pure, phenomodel=None, phenotargets=None):

				per_replica_losses, local_num_total, local_num_correct, local_num_total_k_mer, local_num_correct_k_mer  = strategy.run(run_optimization, args=(model, model2, optimizer, optimizer2, loss_function, input, targets, pure, phenomodel, phenotargets))

				#per_replica_losses, local_num_total, local_num_correct  = strategy.run(train_step, args=(model, model2, optimizer, optimizer2, loss_function, input, targets, pure, phenomodel, phenotargets))

				loss = strategy.reduce("SUM", per_replica_losses, axis=None)
				num_total = strategy.reduce("SUM", local_num_total, axis=None)
				num_correct = strategy.reduce("SUM", local_num_correct, axis=None)
				num_total_k_mer = strategy.reduce("SUM", local_num_total_k_mer, axis=None)
				num_correct_k_mer = strategy.reduce("SUM", local_num_correct_k_mer, axis=None)

				return loss, num_total, num_correct, num_total_k_mer,num_correct_k_mer



			@tf.function
			def train_step(model, model2, optimizer, optimizer2, loss_function, input, targets, pure, phenomodel=None, phenotargets=None):

				with tf.GradientTape() as tape:
					output, _ = model(input)
					loss = loss_function(y_pred = output, y_true = targets)
					loss += tf.nn.scale_regularization_loss(sum(model.losses))

				grads = tape.gradient(loss, model.trainable_variables)
				optimizer.apply_gradients(zip(grads, model.trainable_variables),experimental_aggregate_gradients=False )

				num_total, num_correct = compute_concordance(input, targets, output, data)



				return loss , num_total, num_correct



			@tf.function
			def valid_batch(autoencoder, loss_func, input_valid_batch, targets_valid_batch):
				output_valid_batch, encoded_data_valid_batch = autoencoder(input_valid_batch, is_training = True, regloss=False)

				valid_loss_batch = loss_func(y_pred = output_valid_batch, y_true = targets_valid_batch)
				num_total, num_correct = compute_concordance(input_valid_batch, targets_valid_batch, output_valid_batch, data)
				num_total_k_mer, num_correct_k_mer = compute_k_mer_concordance(input_valid_batch, targets_valid_batch, output_valid_batch, data)

				return valid_loss_batch, num_total, num_correct,num_total_k_mer, num_correct_k_mer

			@tf.function
			def distributed_valid_batch(autoencoder, loss_func, input_valid_batch, targets_valid_batch):

				per_replica_losses, local_num_total, local_num_correct ,local_num_total_k_mer, local_num_correct_k_mer   = strategy.run(valid_batch, args=(autoencoder, loss_func, input_valid_batch, targets_valid_batch))

				loss = strategy.reduce("SUM", per_replica_losses, axis=None)
				num_total = strategy.reduce("SUM", local_num_total, axis=None)
				num_correct = strategy.reduce("SUM", local_num_correct, axis=None)
				num_total_k_mer = strategy.reduce("SUM", local_num_total_k_mer, axis=None)
				num_correct_k_mer = strategy.reduce("SUM", local_num_correct_k_mer, axis=None)

				return loss, num_total, num_correct, num_total_k_mer, num_correct_k_mer

			@tf.function
			def compute_concordance(input, truth, prediction, data):
				"""
				I want to compute the genotype concordance of the prediction when using multi-gpu training.

				Compute them for each batch, and at the end of the epoch just average over the batches?
				Want to only compute the concordance for non-missing data

				"""

				if train_opts["loss"]["class"] == "MeanSquaredError" and (data_opts["norm_mode"] == "smartPCAstyle" or data_opts["norm_mode"] == "standard"):
					try:
						scaler = data.scaler
					except:
						chief_print("Could not calculate predicted genotypes and genotype concordance. No scaler available in data handler.")
						genotypes_output = np.array([])
						true_genotypes = np.array([])

					genotypes_output = to_genotypes_invscale_round(prediction[:, 0:n_markers], scaler_vals = [data.scaler.mean_, data.scaler.var_])
					true_genotypes = to_genotypes_invscale_round(truth, scaler_vals = [data.scaler.mean_, data.scaler.var_])

				elif train_opts["loss"]["class"] == "BinaryCrossentropy" and data_opts["norm_mode"] == "genotypewise01":
					genotypes_output = to_genotypes_sigmoid_round(prediction[:, 0:n_markers])
					true_genotypes = truth

				elif train_opts["loss"]["class"] in ["CategoricalCrossentropy", "KLDivergence"] and data_opts["norm_mode"] == "genotypewise01":
					genotypes_output = tf.cast(tf.argmax(alfreqvector(prediction[:, 0:n_markers]), axis = -1), tf.float32) * 0.5
					true_genotypes = truth

				else:
					chief_print("Could not calculate predicted genotypes and genotype concordance. Not implemented for loss {0} and normalization {1}.".format(train_opts["loss"]["class"],
																																						data_opts["norm_mode"]))
					genotypes_output = np.array([])
					true_genotypes = np.array([])


				use_indices = tf.where(input[:,:, 1] !=  2  )

				diff = true_genotypes - genotypes_output
				diff2 = tf.gather_nd(diff, indices=use_indices)
				num_total = tf.cast(tf.shape(diff2)[0], tf.float32)
				num_correct = tf.cast(tf.shape(tf.where(diff2 == 0))[0], tf.float32)

				return num_total, num_correct

			@tf.function
			def compute_k_mer_concordance(input, truth, prediction, data):
				"""
                I want to compute the genotype concordance of the prediction when using multi-gpu training.

                Compute them for each batch, and at the end of the epoch just average over the batches?
                Want to only compute the concordance for non-missing data

				#TODO This function can not run multi - GPU. It seems like the reduction results in some shape mismatch
                """

				if train_opts["loss"]["class"] == "MeanSquaredError" and (
						data_opts["norm_mode"] == "smartPCAstyle" or data_opts["norm_mode"] == "standard"):
					try:
						scaler = data.scaler
					except:
						chief_print(
							"Could not calculate predicted genotypes and genotype concordance. No scaler available in data handler.")
						genotypes_output = np.array([])
						true_genotypes = np.array([])

					genotypes_output = to_genotypes_invscale_round(prediction[:, 0:n_markers],
																   scaler_vals=[data.scaler.mean_, data.scaler.var_])
					true_genotypes = to_genotypes_invscale_round(truth,
																 scaler_vals=[data.scaler.mean_, data.scaler.var_])

				elif train_opts["loss"]["class"] == "BinaryCrossentropy" and data_opts["norm_mode"] == "genotypewise01":
					genotypes_output = to_genotypes_sigmoid_round(prediction[:, 0:n_markers])
					true_genotypes = truth

				elif train_opts["loss"]["class"] in ["CategoricalCrossentropy", "KLDivergence"] and data_opts[
					"norm_mode"] == "genotypewise01":
					genotypes_output = tf.cast(tf.argmax(alfreqvector(prediction[:, 0:n_markers]), axis=-1),
											   tf.float32) * 0.5
					true_genotypes = truth

				else:
					chief_print(
						"Could not calculate predicted genotypes and genotype concordance. Not implemented for loss {0} and normalization {1}.".format(
							train_opts["loss"]["class"],
							data_opts["norm_mode"]))
					genotypes_output = np.array([])
					true_genotypes = np.array([])

				num_total = []
				num_correct = []
				for k in k_vec:

					truth2 = tf.transpose(
						tf.concat([tf.cast(truth, tf.float32), -1.0 * tf.ones(shape=[tf.shape(truth)[0], 1])], axis=1))
					prediction2 = tf.transpose(
						tf.concat([tf.cast(genotypes_output, tf.float32), -1.0 * tf.ones(shape=[tf.shape(truth)[0], 1])],
								  axis=1))
					d1 = tf.stack([truth2[i:-k + i, :] for i in range(k)], axis=1)
					d2 = tf.stack([prediction2[i:-k + i, :] for i in range(k)], axis=1)

					diff = tf.math.reduce_sum(
						tf.math.reduce_sum(tf.cast(tf.math.reduce_sum(tf.math.abs(d1 - d2), axis=1) == 0, tf.float32),
										   axis=0))


					num_total_temp = tf.cast(tf.shape(d1)[0] * tf.shape(d1)[2],tf.float32)
					num_correct_temp = tf.cast(diff,tf.float32)
					num_total.append(num_total_temp)
					num_correct.append(num_correct_temp)

				num_correct = tf.stack(num_correct, axis=0)
				num_total = tf.stack(num_total, axis=0)


				num_correct = tf.reshape(num_correct, [len(k_vec),1])
				num_total = tf.reshape(num_total,  [len(k_vec),1])

				return num_total, num_correct

			@tf.function
			def distributed_compute_concordance(input, truth, prediction,data):

				local_num_total, local_num_correct = strategy.run(compute_concordance, args = (input, truth, prediction,data))

				num_total = strategy.reduce("SUM", local_num_total, axis=None)
				num_correct = strategy.reduce("SUM", local_num_correct, axis=None)

				return num_total, num_correct



			@tf.function
			def run_optimization(model, model2, optimizer, optimizer2, loss_function, input, targets, pure, phenomodel=None, phenotargets=None):
				'''
				Run one step of optimization process based on the given data.

				:param model: a tf.keras.Model
				:param optimizer: a tf.keras.optimizers
				:param loss_function: a loss function
				:param input: input data
				:param targets: target data
				:return: value of the loss function
				'''
				#val = model.ge.uniform((), minval=0, maxval=1.0)
				#full_loss = val < 0.5
				full_loss = True
				do_two = False
				do_softmaxed = False
				with tf.GradientTape() as g:
					output, encoded_data = model(input, targets, is_training=True, regloss=False)
					if pure and phenomodel is not None:
						z = phenomodel(encoded_data, is_training=True)
						loss_value = tf.reduce_sum(z[0])
					if pure or full_loss:
						loss_value = loss_function(y_pred = output, y_true = targets)
						if do_two:
							output2, _ = model2(input, targets, is_training=True, regloss=False)
							loss_value += loss_function(y_pred = output2, y_true = targets)

					#else:

					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))

				num_total, num_correct = compute_concordance(input, targets, output, data)

				num_total_k_mer, num_correct_k_mer = compute_k_mer_concordance(input, targets, output, data)

				allvars = model.trainable_variables + (model2.trainable_variables if model2 is not None else []) + (phenomodel.trainable_variables if phenomodel is not None else [])
				#print("ALLVARS", allvars, "###")
				gradients = g.gradient(loss_value, allvars)

				orig_loss = loss_value
				with tf.GradientTape() as g5:
					loss_value = tf.constant(0.)
					if do_softmaxed:
						for output, encoded_data in (model(input, targets, is_training=True, regloss=False),) + ((model2(input, targets, is_training=True, regloss=False), ) if do_two else ()):
							y_true = tf.one_hot(tf.cast(targets * 2, tf.uint8), 3)
							y_pred = tf.nn.softmax(output[:,0:model.n_markers])
							#0*tf.math.reduce_mean(y_pred,axis=0,keepdims=True)
							loss_value += tf.math.reduce_sum(((-y_pred) * y_true)) * 1e-6
					for val in allvars:
						maxflax = tf.math.reduce_max(tf.math.abs(val))
						maxflax2 = tf.math.minimum(tf.math.reduce_max(tf.math.abs(val)), tf.math.reduce_max(tf.math.abs(1. - tf.math.abs(val))))
						#tf.print("MAX", maxflax)
						loss_value += tf.square(tf.math.maximum(1.0, maxflax2))
					#if pure or full_loss:
					#	loss_value = -loss_function(y_pred = output, y_true = targets, avg=True)

					#else:

					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))

				gradientsavg = g5.gradient(loss_value, allvars)
				other_loss4 = loss_value
				#other_loss4 = 0

				#with tf.GradientTape() as g4:
				#	output, encoded_data = model(input, targets, is_training=True, rander=[False, True])
				#	if pure or full_loss:
				#		loss_value = loss_function(y_pred = output, y_true = targets)

					#else:

					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))

				#gradientsrandx = g4.gradient(loss_value, model.trainable_variables)
				#other_loss3 = loss_value
				#with tf.GradientTape() as g4:
				#	output, encoded_data = model(input, targets, is_training=True)
				#	if pure or full_loss:
				#		loss_value = loss_function(y_pred = output, y_true = targets, pow=2.)

					#else:

					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))

				#gradientssq = g4.gradient(loss_value, model.trainable_variables)
				#other_loss3 = loss_value

				with tf.GradientTape() as g4:
					output, encoded_data = model(input, targets, is_training=True)

					#loss_value = tf.math.reduce_sum(tf.reduce_sum(tf.square(encoded_data-encoded_data2),axis=-1))*1e-2
					#loss_value = loss_function(y_pred = output, y_true = targets) * (1.0 if pure or full_loss else 0.0)
					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
					#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))
					loss_value = sum(model.losses)
					if do_two:
						output2, encoded_data2 = model2(input, targets, is_training=True)
						loss_value += sum(model2.losses)
				gradientsc = g4.gradient(loss_value, allvars)
				other_loss2 = loss_value

				if do_two:
					factor = 0.
					with tf.GradientTape() as g2:
						output, encoded_data = model(input, targets, is_training=True, regloss=False)
						output2, encoded_data2 = model2(input, targets, is_training=True, regloss=False)
						loss_value = tf.math.reduce_sum( -tf.math.log(0.5+0.5*tf.reduce_sum((factor*encoded_data-tf.roll(encoded_data, 1, axis=0))
						* (factor*encoded_data2-tf.roll(encoded_data2, 1, axis=0)), axis=-1)
						* tf.math.rsqrt
						(tf.reduce_sum((factor*encoded_data-tf.roll(encoded_data, 1, axis=0)) * (factor*encoded_data-tf.roll(encoded_data, 1, axis=0)), axis=-1) * tf.reduce_sum(
						(factor*encoded_data2-tf.roll(encoded_data2, 1, axis=0)) * (factor*encoded_data2-tf.roll(encoded_data2, 1, axis=0))+1e-4, axis=-1))))*1e-2
					gradients2 = g2.gradient(loss_value, allvars)
					other_loss = loss_value

					with tf.GradientTape() as g3:
						output, encoded_data = model(input, targets, is_training=True, regloss=False)
						output2, encoded_data2 = model2(input, targets, is_training=True, regloss=False)
						loss_value = tf.math.reduce_sum( -tf.math.log(1.-0.5*tf.reduce_sum((factor*encoded_data-tf.roll(encoded_data, 1, axis=0))
						* (factor*encoded_data2-tf.roll(encoded_data2, 2, axis=0)), axis=-1)
						* tf.math.rsqrt
						(tf.reduce_sum((factor*encoded_data-tf.roll(encoded_data, 1, axis=0)) * (factor*encoded_data-tf.roll(encoded_data, 1, axis=0)), axis=-1) * tf.reduce_sum(
						(factor*encoded_data2-tf.roll(encoded_data2, 2, axis=0)) * (factor*encoded_data2-tf.roll(encoded_data2, 2, axis=0))+1e-4, axis=-1))))*1e-2
					##	output, encoded_data = model(input, targets, is_training=True)
					##	#loss_value = loss_function(y_pred = output, y_true = targets) * (1.0 if pure or full_loss else 0.0)
						#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "AD_066", tf.math.reduce_sum(tf.square(encoded_data), axis=-1), 0.))
						#loss_value += 1e-3*tf.reduce_sum(tf.where(poplist[:, 0] == "Zapo0097", tf.square(encoded_data[:, 1]) + tf.square(tf.minimum(0.0, encoded_data[:, 0] - 1.0)), 0.))
					##	y_pred = output - tf.stop_gradient(tf.math.reduce_max(output, axis=-1, keepdims=True))
					##	loss_value = -tf.math.reduce_mean(tf.square(y_pred - tf.roll(y_pred, 1, axis = 0)) * 1e-3)
					gradientsb = g3.gradient(loss_value, allvars)
					other_loss3 = loss_value

				if phenomodel is not None:
					with tf.GradientTape() as g6:
						loss_value = tf.constant(0.)
						for output, encoded_data in (model(input, targets, is_training=True, regloss=False),) + ((model2(input, targets, is_training=True, regloss=False), ) if do_two else ()):
							phenopred, _ = phenomodel(encoded_data, is_training=True)
							tf.print("PRED")
							tf.print(phenopred)
							tf.print(phenotargets)
							loss_value += tf.math.reduce_sum(tf.square(phenopred - phenotargets)) * 1e-2

					gradientspheno = g6.gradient(loss_value, allvars)
					phenoloss = loss_value


				loss_value = orig_loss
				##radients3 = []

				##loss_value += orig_loss

				##for g1, g2 in zip(gradients, gradients2):
				##	if g1 is None:
				##		g3 = g2
				##	elif g2 is None:
				##		g3 = g1
				##	else:
				##		#g3 = tf.where(tf.math.sign(g1 * g2) >= 0, g1 + g2, 0.)
				##		summed = g1 + g2
				##		g3 = tf.where(tf.math.sign(g1 * g2) >= 0, summed, tf.math.sign(summed) * tf.math.minimum(0.5 * tf.math.minimum(tf.abs(g1), tf.abs(g2)), tf.abs(summed)))
				##	gradients3.append(g3)
				def combine(gradients, gradients2):
					alphanom = tf.constant(0.)
					alphadenom = tf.constant(1.0e-30)
					for g1, g2 in zip(gradients, gradients2):
						if g1 is not None and g2 is not None:
							gdiff = g2 - g1
							alphanom += tf.math.reduce_sum(gdiff * g2)
							alphadenom += tf.math.reduce_sum(gdiff * gdiff)
					alpha = alphanom / alphadenom
					gradients3 = []
					cappedalpha = tf.clip_by_value(alpha, 0., 1.)
					for g1, g2 in zip(gradients, gradients2):
						if g1 is None:
							gradients3.append(g2)
						elif g2 is None:
							gradients3.append(g1)
						else:
							gradients3.append(g1 * (1-cappedalpha) + g2 * (cappedalpha))
					return (gradients3, alpha)

				#gradients4, alpha4 = combine(gradientsrandx, gradientsrandy)
				#alpha4 = 0
				gradients3, alpha4 = combine(gradients, gradientsavg)
				alpha3 = 0
				#gradients4 = gradientsrandx
				if do_two:
					gradients4, alpha3 = combine(gradients2, gradientsb)
					gradients3, alpha = combine(gradients3, gradients4)
				else:
					alpha3 = 0
					alpha = 0
					other_loss3 = 0
					other_loss = 0

				gradients3, alpha2 = combine(gradients3, gradientsc)
				if phenomodel is not None:
					gradients3, phenoalpha = combine(gradients3, gradientspheno)
				else:
					phenoloss, phenoalpha = (0.,0.)
				if pure or full_loss:
					optimizer.apply_gradients(zip(gradients3, allvars))
				#if pure or not full_loss:
				#	# was optimizer2
				#	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
				#tf.print(loss_value, other_loss4, other_loss3, other_loss2, other_loss, full_loss, alpha4, alpha3, alpha2, alpha)
				#tf.print(loss_value, other_loss4, other_loss3, other_loss2, other_loss, phenoloss, full_loss, alpha4, alpha3, alpha2, alpha, phenoalpha)
				return loss_value, num_total, num_correct, num_total_k_mer, num_correct_k_mer



	if arguments['train']:

		try:
			resume_from = int(arguments["resume_from"])
			if resume_from < 1:
				saved_epochs = get_saved_epochs(train_directory)
				resume_from = saved_epochs[-1]
		except:
			resume_from = False




		epochs = int(arguments["epochs"])
		save_interval = int(arguments["save_interval"])

		data = alt_data_generator(filebase= data_prefix,
						batch_size = batch_size,
						normalization_mode = norm_mode,
						normalization_options = norm_opts,
						impute_missing = fill_missing,
						sparsifies  = sparsifies,
						recombination_rate= recomb_rate)

		tf.print("Recombination rate: ", data.recombination_rate)

		n_markers = data.n_markers
		if holdout_val_pop is None:
			data.define_validation_set2(validation_split= validation_split)
		else:
			data.define_validation_set_holdout_pop(holdout_pop= holdout_val_pop,superpopulations_file=superpopulations_file)
			superpops = pd.read_csv(superpopulations_file, header=None).to_numpy()

		data.missing_mask_input = missing_mask_input

		n_unique_train_samples = copy.deepcopy(data.n_train_samples)
		n_valid_samples = copy.deepcopy(data.n_valid_samples)


		if "n_samples" in train_opts.keys() and int(train_opts["n_samples"]) > 0:
			n_train_samples = int(train_opts["n_samples"])
		else:
			n_train_samples = n_unique_train_samples

		batch_size_valid = batch_size
		n_train_batches, n_train_samples_last_batch = get_batches(n_train_samples, batch_size)
		n_valid_batches, n_valid_samples_last_batch = get_batches(n_valid_samples, batch_size_valid)
		data.n_train_samples_last_batch = n_train_samples_last_batch
		data.n_valid_samples_last_batch = n_valid_samples_last_batch

		train_times = []
		train_epochs = []
		save_epochs = []

		############### setup learning rate schedule ##############
		step_counter = resume_from * n_train_batches
		if "lr_scheme" in train_opts.keys():
			schedule_module = getattr(eval(train_opts["lr_scheme"]["module"]), train_opts["lr_scheme"]["class"])
			schedule_args = train_opts["lr_scheme"]["args"]

			if "decay_every" in schedule_args:
				decay_every = int(schedule_args.pop("decay_every"))
				decay_steps = n_train_batches * decay_every
				schedule_args["decay_steps"] = decay_steps

			lr_schedule = schedule_module(learning_rate, **schedule_args)

			# use the schedule to calculate what the lr was at the epoch were resuming from
			updated_lr = lr_schedule(step_counter)
			lr_schedule = schedule_module(updated_lr, **schedule_args)

			chief_print("Using learning rate schedule {0}.{1} with {2}".format(train_opts["lr_scheme"]["module"], train_opts["lr_scheme"]["class"], schedule_args))
		else:
			lr_schedule = False

		chief_print("\n______________________________ Data ______________________________")
		chief_print("N unique train samples: {0}".format(n_unique_train_samples))
		chief_print("--- training on : {0}".format(n_train_samples))
		chief_print("N valid samples: {0}".format(n_valid_samples))
		chief_print("N markers: {0}".format(n_markers))
		chief_print("")



		chief_print("\n______________________________ Train ______________________________")
		chief_print("Model layers and dimensions:")
		chief_print("-----------------------------")

		chunk_size = 5 * data.batch_size


		ds = data.create_dataset(chunk_size, "training")
		dds = strategy.experimental_distribute_dataset(ds)
		ds_validation = data.create_dataset(chunk_size, "validation")
		dds_validation = strategy.experimental_distribute_dataset(ds_validation)


		with strategy.scope():
				# Initialize the model and optimizer


			autoencoder = Autoencoder(model_architecture, n_markers, noise_std, regularizer)
			autoencoder2 = None #  Autoencoder(model_architecture, n_markers, noise_std, regularizer)

			if pheno_model_architecture is not None:
				pheno_model = Autoencoder(pheno_model_architecture, 2, noise_std, regularizer)
			else:
				pheno_model = None
			optimizer = tf.optimizers.Adam(learning_rate = lr_schedule, beta_1=0.99, beta_2 = 0.999)
			optimizer2 = tf.optimizers.Adam(learning_rate = lr_schedule, beta_1=0.99, beta_2 = 0.999)


			input_test, targets_test, _ = next(ds.as_numpy_iterator())
			#input_test2 = np.zeros(shape = (2,n_markers,2))
			output_test, encoded_alt_data_generator = autoencoder(input_test, is_training = False, verbose = True)

			if resume_from:
				chief_print("\n______________________________ Resuming training from epoch {0} ______________________________".format(resume_from))
				weights_file_prefix = "{0}/{1}/{2}".format(train_directory, "weights", resume_from)
				chief_print("Reading weights from {0}".format(weights_file_prefix))
				autoencoder.load_weights(weights_file_prefix)


				input_test, targets_test, _ = next(ds.as_numpy_iterator())

				output_test, encoded_alt_data_generator = autoencoder(input_test[0:2,:,:], is_training = False, verbose = True)

		######### Create objects for tensorboard summary ###############################
		if isChief:
			train_writer = tf.summary.create_file_writer(train_directory + '/train')
			valid_writer = tf.summary.create_file_writer(train_directory + '/valid')
		######################################################

		tf.print(autoencoder.summary())
		# train losses per epoch
		losses_t = []
		conc_t = []
		# valid losses per epoch
		losses_v = []
		conc_v = []

		""""
		Enable the below command to log the runs in the profiler, also de-comment the line after the training loop. 

		Used for debugging the code / profiling different runs. Visualized in TensorBoard. I like to use ThinLinc to look at it, seems like the
		least effort solution to get tensorboard up and running. Possible to do it in other ways, but more painful
		"""
		suffix = ""
		#if slurm_job:
		#	suffix = str(os.environ["SLURMD_NODENAME"])
		logs = train_directory+ "/logdir/"  + datetime.now().strftime("%Y%m%d-%H%M%S") + suffix

		profile = 0

		for e in range(1,epochs+1):

				if e % 100 < 50:
					autoencoder.passthrough.assign(0.0)
				else:
					autoencoder.passthrough.assign(1.0)
				if e ==2:

					if profile : tf.profiler.experimental.start(logs)
				startTime = datetime.now()
				effective_epoch = e + resume_from

				train_loss = 0
				conc_total = 0
				conc_correct = 0
				for batch_dist_input, batch_dist_target, poplist in dds:

					pt = generatepheno(phenodata, poplist)
					train_batch_loss, num_total, num_correct, num_total_k_mer,num_correct_k_mer = distributed_train_step(autoencoder, autoencoder2, optimizer, optimizer2, loss_func, batch_dist_input, batch_dist_target, False, phenomodel=pheno_model, phenotargets=pt)

					if holdout_val_pop is not None and num_devices == 1 and num_workers == 1 :

						sample_superpop = np.array([superpops[np.where(poplist[i,1] == superpops[:, 0])[0][0], 1] for i in range(len(poplist[:,1]))])
						assert(np.sum(np.where(sample_superpop == "holdout_val_pop")[0]) == 0 )

					train_loss += train_batch_loss
					conc_total += num_total
					conc_correct += num_correct
				train_loss_this_epoch = train_loss / (n_train_samples*0  + n_train_batches* (data.total_batch_size -data.batch_size))
				train_conc_this_epoch = num_correct / num_total
				train_conc_this_epoch_k_mer = num_correct_k_mer / num_total_k_mer

				if isChief:
					with train_writer.as_default():
						tf.summary.scalar('loss', train_loss_this_epoch, step = step_counter)
						if lr_schedule:
							tf.summary.scalar("learning_rate", optimizer._decayed_lr(var_dtype=tf.float32), step = step_counter)
						else:
							tf.summary.scalar("learning_rate", learning_rate, step = step_counter)

				train_time = (datetime.now() - startTime).total_seconds()
				train_times.append(train_time)
				train_epochs.append(effective_epoch)
				losses_t.append(train_loss_this_epoch)
				conc_t.append(train_conc_this_epoch)

				if e == 1:
					conc_t_k_mer = tf.reshape(train_conc_this_epoch_k_mer, [1, 5])
				else:
					conc_t_k_mer = tf.concat([conc_t_k_mer, tf.reshape(train_conc_this_epoch_k_mer, [1, 5])], axis=0)

				chief_print("Epoch: {}/{}...".format(effective_epoch, epochs+resume_from))
				chief_print("--- Train loss: {:.4f} Concordance {:.4f} time: {}".format(train_loss_this_epoch, train_conc_this_epoch_k_mer[0,0],  train_time))

				if n_valid_samples > 0:

					startTime = datetime.now()
					valid_loss = 0
					conc_valid_total = 0
					conc_valid_correct = 0
					conc_valid_total_k_mer = 0
					conc_valid_correct_k_mer = 0
					for input_valid_batch, targets_valid_batch, poplist  in dds_validation:

						valid_loss_batch, num_total, num_correct, num_total_k_mer, num_correct_k_mer  = distributed_valid_batch(autoencoder, loss_func, input_valid_batch, targets_valid_batch)

						valid_loss += valid_loss_batch
						conc_valid_total += num_total
						conc_valid_correct += num_correct

						conc_valid_total_k_mer += num_total_k_mer
						conc_valid_correct_k_mer += num_correct_k_mer

						if holdout_val_pop is not None and num_devices == 1 and num_workers == 1 :
							sample_superpop = np.array([superpops[np.where(poplist[i, 1] == superpops[:, 0])[0][0], 1] for i in range(len(poplist[:, 1]))])
							#tf.print(sample_superpop)

							assert (np.sum(np.where(sample_superpop != holdout_val_pop)[0]) == 0)

					valid_conc_this_epoch = conc_valid_correct / conc_valid_total
					valid_conc_this_epoch_k_mer = conc_valid_correct_k_mer / conc_valid_total_k_mer

					valid_loss_this_epoch = valid_loss  / n_valid_samples

					if isChief:
						with valid_writer.as_default():
							tf.summary.scalar('loss', valid_loss_this_epoch, step=step_counter)

					losses_v.append(valid_loss_this_epoch)
					conc_v.append(valid_conc_this_epoch)
					if e == 1:
						conc_v_k_mer = tf.reshape(valid_conc_this_epoch_k_mer, [1,5])
					else:
						conc_v_k_mer = tf.concat([conc_v_k_mer, tf.reshape(valid_conc_this_epoch_k_mer, [1,5])], axis = 0)

					valid_time = (datetime.now() - startTime).total_seconds()
					chief_print("--- Valid loss: {:.4f} Concordance {:.4f} time: {}".format(valid_loss_this_epoch, valid_conc_this_epoch_k_mer[0,0], valid_time))

				if isChief:
					weights_file_prefix = train_directory + "/weights/" + str(effective_epoch)
					pheno_weights_file_prefix = train_directory + "/pheno_weights/" + str(effective_epoch)
					if e % save_interval == 0:
						startTime = datetime.now()
						save_weights(train_directory, weights_file_prefix, autoencoder)
						save_weights(train_directory, pheno_weights_file_prefix, pheno_model)
						save_time = (datetime.now() - startTime).total_seconds()
						save_epochs.append(effective_epoch)
						print("-------- Save time: {0} dir: {1}".format(save_time, weights_file_prefix))
		if profile: tf.profiler.experimental.stop()

		if isChief:
			outfilename = train_directory + "/" + "train_times.csv"
			write_metric_per_epoch_to_csv(outfilename, train_times, train_epochs)

			outfilename = "{0}/losses_from_train_t.csv".format(train_directory)
			epochs_t_combined, losses_t_combined = write_metric_per_epoch_to_csv(outfilename, losses_t, train_epochs)
			fig, ax = plt.subplots()
			plt.plot(epochs_t_combined, losses_t_combined, label="train", c="orange")

			if n_valid_samples > 0:
				outfilename = "{0}/losses_from_train_v.csv".format(train_directory)
				epochs_v_combined, losses_v_combined = write_metric_per_epoch_to_csv(outfilename, losses_v, train_epochs)
				plt.plot(epochs_v_combined, losses_v_combined, label="valid", c="blue")
				min_valid_loss_epoch = epochs_v_combined[np.argmin(losses_v_combined)]
				min_valid_loss = np.min(losses_v_combined)
				plt.axvline(min_valid_loss_epoch, color="black")
				plt.text(min_valid_loss_epoch + 0.1, 0.5,'min valid loss at epoch {}'.format(int(min_valid_loss_epoch)),
						rotation=90,
						transform=ax.get_xaxis_text1_transform(0)[0])
				plt.title(" Min Valid loss: {:.4f}".format(min_valid_loss))
			plt.xlabel("Epoch")
			plt.ylabel("Loss function value")
			plt.legend()
			plt.savefig("{}/losses_from_train.pdf".format(train_directory))
			plt.close()



			plt.figure()
			plt.plot(conc_t, label = "Training", linewidth = 2)
			plt.plot(conc_v, label = "Validation",  linewidth = 2)
			plt.plot(np.ones(len(conc_t)) * data.baseline_concordance, 'k',  linewidth = 2, label = "Baseline")
			plt.legend()
			plt.xlabel("Epoch")
			plt.ylabel("Genotype Concordance")
			plt.savefig("{}/concordances.pdf".format(train_directory))
			plt.close()

			plt.figure()
			for i in range(conc_v_k_mer.shape[1]):
				plt.plot(conc_v_k_mer.numpy()[:, i],color = "C{}".format(i), label="Validation {}-mer".format(k_vec[i]), linewidth=2)
				plt.plot(np.ones(len(conc_t)) * data.baseline_concordances_k_mer[i], linestyle='dashed',color = "C{}".format(i), linewidth=2, label="Baseline{}-mer".format(k_vec[i]))
				outfilename = "{0}/conc_v_{1}_mer.csv".format(train_directory,k_vec[i])

				epochs_combined, genotype_concs_combined = write_metric_per_epoch_to_csv(outfilename,
																					conc_v_k_mer.numpy()[:, i], train_epochs)

			#plt.plot(np.ones(len(conc_t)) * data.baseline_concordance, 'k', linewidth=2, label="Baseline")
			plt.legend(prop ={"size":3} )
			plt.xlabel("Epoch")
			plt.ylabel("Genotype Concordance")
			plt.savefig("{}/concordances_k_meres.pdf".format(train_directory))
			plt.close()

			plt.figure()
			for i in range(conc_v_k_mer.shape[1]):
				plt.plot(conc_v_k_mer.numpy()[:, i], color="C{}".format(i), label="Validation {}-mer".format(k_vec[i]),
						 linewidth=2)
				plt.plot(conc_t_k_mer.numpy()[:, i], linestyle='dotted',color="C{}".format(i), label="training {}-mer".format(k_vec[i]),
						 linewidth=2)
				plt.plot(np.ones(len(conc_t)) * data.baseline_concordances_k_mer[i], linestyle='dashed',
						 color="C{}".format(i), linewidth=2, label="Baseline{}-mer".format(k_vec[i]))
				outfilename = "{0}/conc_t_{1}_mer.csv".format(train_directory, k_vec[i])

				epochs_combined, genotype_concs_combined = write_metric_per_epoch_to_csv(outfilename,
																						 conc_t_k_mer.numpy()[:, i],
																						 train_epochs)

			# plt.plot(np.ones(len(conc_t)) * data.baseline_concordance, 'k', linewidth=2, label="Baseline")
			plt.legend(prop={"size": 3})
			plt.xlabel("Epoch")
			plt.ylabel("Genotype Concordance")
			plt.savefig("{}/concordances_both_k_meres.pdf".format(train_directory))
			plt.close()



			tf.print(conc_v_k_mer.numpy()[-1,:]/ data.baseline_concordances_k_mer)
		chief_print("Done training. Wrote to {0}".format(train_directory))



	if arguments['project'] and isChief:

		projected_epochs = get_projected_epochs(encoded_data_file)

		if arguments['epoch']:
			epoch = int(arguments['epoch'])
			epochs = [epoch]

		else:
			epochs = get_saved_epochs(train_directory)

		#for projected_epoch in projected_epochs:
		#	if alt_data is None:
		#		try:
	#				epochs.remove(projected_epoch)
		#		except:
		#			continue

		chief_print("Projecting epochs: {0}".format(epochs))
		chief_print("Already projected: {0}".format(projected_epochs))

		# Make this larger. For some reason a low project batch size resulted in division by zero in the normalization, which yields a bad scaler
		batch_size_project = batch_size
		sparsify_fraction = 0.0


		data.sparsifies = [sparsify_fraction]
		if alt_data is not None:
			data_prefix = datadir+alt_data

		data = alt_data_generator(filebase= data_prefix,
				batch_size = batch_size_project,
				normalization_mode = norm_mode,
				normalization_options = norm_opts,
				impute_missing = fill_missing)
		data._define_samples() # This sets the number of validation samples to be 0
		ind_pop_list_train_reference = data.ind_pop_list_train_orig[data.sample_idx_train]

		write_h5(encoded_data_file, "ind_pop_list_train", np.array(ind_pop_list_train_reference, dtype='S'))

		#####################

		n_train_samples = copy.deepcopy(data.n_train_samples)
		chief_print("n_train_samples: " + str(n_train_samples))

		n_train_batches, n_train_samples_last_batch = get_batches(n_train_samples, batch_size_project)
		n_valid_samples = 0
		chief_print("n_train_samples_last_batch: " + str(n_train_samples_last_batch))
		batch_size_valid = 1
		n_valid_batches, n_valid_samples_last_batch = get_batches(n_valid_samples, batch_size_valid)

		data.n_valid_samples_last_batch = n_valid_samples_last_batch
		data.n_train_samples_last_batch = n_train_samples_last_batch

		data.missing_mask_input = missing_mask_input



		############################
			# Here, create new dataset, with the same train split as in the training step

		data_project = alt_data_generator(filebase= data_prefix,
						batch_size = batch_size,
						normalization_mode = norm_mode,
						normalization_options = norm_opts,
						impute_missing = fill_missing,
						sparsifies  = sparsifies,
						recombination_rate= recomb_rate)
		n_markers = data_project.n_markers
		data_project.sparsifies = [sparsify_fraction]

		if holdout_val_pop is None:
			data_project.define_validation_set2(validation_split= validation_split)
		else:
			data_project.define_validation_set_holdout_pop(holdout_pop= holdout_val_pop,superpopulations_file=superpopulations_file)
			superpops = pd.read_csv(superpopulations_file, header=None).to_numpy()

		superpops = pd.read_csv(superpopulations_file, header=None).to_numpy()

		data_project.missing_mask_input = missing_mask_input

		n_unique_train_samples = copy.deepcopy(data_project.n_train_samples)
		n_valid_samples = copy.deepcopy(data_project.n_valid_samples)


		if "n_samples" in train_opts.keys() and int(train_opts["n_samples"]) > 0:
			n_train_samples = int(train_opts["n_samples"])
		else:
			n_train_samples = n_unique_train_samples

		batch_size_valid = batch_size
		n_train_batches, data_project.n_train_samples_last_batch  = get_batches(n_train_samples, batch_size)
		n_valid_batches, data_project.n_valid_samples_last_batch = get_batches(n_valid_samples, batch_size_valid)

		chunk_size = 5 * data_project.batch_size

		ds_project =            data_project.create_dataset(chunk_size, "training")
		ds_validation_project = data_project.create_dataset(chunk_size, "validation")
		#####################

		# loss function of the train set per epoch
		losses_train = []

		# genotype concordance of the train set per epoch
		genotype_concs_train = []
		genotype_concordance_metric = GenotypeConcordance()

		autoencoder = Autoencoder(model_architecture, n_markers, noise_std, regularizer)
		if pheno_model_architecture is not None:
			pheno_model = Autoencoder(pheno_model_architecture, 2, noise_std, regularizer)
		else:
			pheno_model = None
			pheno_train = None


		genotype_concordance_metric = GenotypeConcordance()

		scatter_points_per_epoch = []
		colors_per_epoch = []
		markers_per_epoch = []
		edgecolors_per_epoch = []

		data.batch_size = batch_size_project
		chunk_size = 5 * data.batch_size
		pheno_train = None
		# HERE WE NEED TO "NOT SHUFFLE" THE DATASET, IN ORDER TO GET EVALUATE TO WORK AS INTENDED (otherwise, there is a problem with the ordering, works on its own,
		# but works differently when directly compared to original implementation)
		ds = data.create_dataset(chunk_size, "training", shuffle = False)


		for epoch in epochs:
			chief_print("########################### epoch {0} ###########################".format(epoch))
			weights_file_prefix = "{0}/{1}/{2}".format(train_directory, "weights", epoch)
			weights_dir = "{0}/{1}".format(train_directory, "weights")
			chief_print("Reading weights from {0}".format(weights_file_prefix))

			autoencoder.load_weights(weights_file_prefix)
			if pheno_model is not None:
				pheno_weights_file_prefix = "{0}/{1}/{2}".format(train_directory, "pheno_weights", epoch)
				pheno_model.load_weights(pheno_weights_file_prefix)


			ind_pop_list_train = np.empty((0,2))
			encoded_train = np.empty((0, n_latent_dim))
			decoded_train = None
			targets_train = np.empty((0, n_markers))

			loss_value_per_train_batch = []
			genotype_conc_per_train_batch = []
			loss_train_batch = 0

			for input_train_batch, targets_train_batch, ind_pop_list_train_batch in ds:
				decoded_train_batch, encoded_train_batch = autoencoder(input_train_batch, is_training = False)
				loss_train_batch = loss_func(y_pred=decoded_train_batch, y_true=targets_train_batch)
				encoded_train = np.concatenate((encoded_train, encoded_train_batch), axis=0)

				if decoded_train is None:
					decoded_train = np.copy(decoded_train_batch[:,0:n_markers])
				else:
					decoded_train = np.concatenate((decoded_train, decoded_train_batch[:,0:n_markers]), axis=0)

				ind_pop_list_train = np.concatenate((ind_pop_list_train, np.array(ind_pop_list_train_batch,dtype = "U25")), axis=0)
				targets_train = np.concatenate((targets_train, targets_train_batch[:,0:n_markers]), axis=0)
				loss_value_per_train_batch.append(loss_train_batch)

			ind_pop_list_train = np.array(ind_pop_list_train)
			encoded_train = np.array(encoded_train)

			list(ind_pop_list_train[:,1])
			list(ind_pop_list_train_reference[:,1])
			loss_value = np.sum(loss_value_per_train_batch)  / n_train_samples # /num_devices

			if epoch == epochs[0]:
				assert len(ind_pop_list_train) == data.n_train_samples, "{0} vs {1}".format(len(ind_pop_list_train), data.n_train_samples)
				assert len(encoded_train) == data.n_train_samples, "{0} vs {1}".format(len(encoded_train), data.n_train_samples)
				assert (list(ind_pop_list_train[:,0])) == (list(ind_pop_list_train_reference[:,0]))
				assert (list(ind_pop_list_train[:,1])) == (list(ind_pop_list_train_reference[:,1]))


			if not fill_missing:
				orig_nonmissing_mask = get_originally_nonmissing_mask(targets_train)
			else:
				orig_nonmissing_mask = np.full(targets_train.shape, True)

			if train_opts["loss"]["class"] == "MeanSquaredError" and (data_opts["norm_mode"] == "smartPCAstyle" or data_opts["norm_mode"] == "standard"):
				try:
					scaler = data.scaler
				except:
					chief_print("Could not calculate predicted genotypes and genotype concordance. No scaler available in data handler.")
					genotypes_output = np.array([])
					true_genotypes = np.array([])

				genotypes_output = to_genotypes_invscale_round(decoded_train[:, 0:n_markers], scaler_vals = [data.scaler.mean_, data.scaler.var_])
				true_genotypes = to_genotypes_invscale_round(targets_train, scaler_vals = [data.scaler.mean_, data.scaler.var_])
				genotype_concordance_metric.update_state(y_pred = genotypes_output[orig_nonmissing_mask],
														 y_true = true_genotypes[orig_nonmissing_mask])


			elif train_opts["loss"]["class"] == "BinaryCrossentropy" and data_opts["norm_mode"] == "genotypewise01":
				genotypes_output = to_genotypes_sigmoid_round(decoded_train[:, 0:n_markers])
				true_genotypes = targets_train
				genotype_concordance_metric.update_state(y_pred = genotypes_output[orig_nonmissing_mask], y_true = true_genotypes[orig_nonmissing_mask])

			elif train_opts["loss"]["class"] in ["CategoricalCrossentropy", "KLDivergence"] and data_opts["norm_mode"] == "genotypewise01":
				if alt_data is None:
					genotypes_output = tf.cast(tf.argmax(alfreqvector(decoded_train[:, 0:n_markers]), axis = -1), tf.float32) * 0.5
					true_genotypes = targets_train
					genotype_concordance_metric.update_state(y_pred = genotypes_output[orig_nonmissing_mask], y_true = true_genotypes[orig_nonmissing_mask])

			else:
				chief_print("Could not calculate predicted genotypes and genotype concordance. Not implemented for loss {0} and normalization {1}.".format(train_opts["loss"]["class"],
																																					data_opts["norm_mode"]))
				genotypes_output = np.array([])
				true_genotypes = np.array([])

			genotype_concordance_value = genotype_concordance_metric.result()

			losses_train.append(loss_value)
			genotype_concs_train.append(genotype_concordance_value)

			if superpopulations_file:
				coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

				if doing_clustering:
					plot_clusters_by_superpop(coords_by_pop, "{0}/clusters_e_{1}".format(results_directory, epoch), superpopulations_file, write_legend = epoch == epochs[0])
				else:
					scatter_points, colors, markers, edgecolors = \
						plot_coords_by_superpop(coords_by_pop,"{0}/dimred_e_{1}_by_superpop".format(results_directory, epoch), superpopulations_file, plot_legend = epoch == epochs[0])

					scatter_points_per_epoch.append(scatter_points)
					colors_per_epoch.append(colors)
					markers_per_epoch.append(markers)
					edgecolors_per_epoch.append(edgecolors)

			else:
				try:
					coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)
					plot_coords_by_pop(coords_by_pop, "{0}/dimred_e_{1}_by_pop".format(results_directory, epoch))
				except:
					plot_coords(encoded_train, "{0}/dimred_e_{1}".format(results_directory, epoch))

			if pheno_train is not None:
				writephenos(f'{results_directory}/{epoch}.phe', ind_pop_list_train, pheno_train)
			write_h5(encoded_data_file, "{0}_encoded_train".format(epoch), encoded_train)


			# For plotting the 2d visualization in the train-test case.
			fsize = 3.3
			markersize = 5


			plt.figure(figsize = (fsize,fsize)) #

			
			for input_valid_batch_proj, targets_valid_batch_proj, ind_pop_list_valid_batch_proj in ds_validation_project:
				decoded_valid_batch_proj, encoded_valid_batch_proj=autoencoder(input_valid_batch_proj,is_training=False)
				plt.plot(encoded_valid_batch_proj[:,0],encoded_valid_batch_proj[:,1], "b1", markersize = markersize)

			for input_train_batch_proj, targets_train_batch_proj, ind_pop_list_train_batch_proj in ds_project:
				decoded_train_batch_proj, encoded_train_batch_proj=autoencoder(input_train_batch_proj, is_training = False)
				plt.plot(encoded_train_batch_proj[:,0], encoded_train_batch_proj[:,1], "r2", markersize = markersize)

			

			plt.plot(encoded_valid_batch_proj[-1,0],encoded_valid_batch_proj[-1,1], "b1", label="Validation", markersize = markersize)
			plt.plot(encoded_train_batch_proj[-1,0], encoded_train_batch_proj[-1,1], "r2", label="Training", markersize = markersize)
			plt.legend(prop={'size': 6})
			plt.tight_layout()
			plt.savefig("{0}/dimred_e_{1}_by_train_test.pdf".format(results_directory, epoch))
			plt.clf()
		try:
			plot_genotype_hist(np.array(genotypes_output), "{0}/{1}_e{2}".format(results_directory, "output_as_genotypes", epoch))
			plot_genotype_hist(np.array(true_genotypes), "{0}/{1}".format(results_directory, "true_genotypes"))
		except:
			pass


		################################################################

		############################### losses ##############################

		outfilename = "{0}/losses_from_project.csv".format(results_directory)
		epochs_combined, losses_train_combined = write_metric_per_epoch_to_csv(outfilename, losses_train, epochs)


		plt.plot(epochs_combined, losses_train_combined,
				 label="all data",
				 c="red")

		plt.xlabel("Epoch")
		plt.ylabel("Loss function value")
		plt.legend()
		plt.savefig(results_directory + "/" + "losses_from_project.pdf")
		plt.close()

		if 'encoded_train' in locals():
			#df = pd.read_parquet(filebase + ".parquet", columns=["0","1"])

			#df_numpy = pd.DataFrame.to_numpy(df)

			#print(df_numpy)
			#out, enc_orig = autoencoder(df_numpy)

			fig, ax = plt.subplots()

			for i in range(encoded_train.shape[0]):

				circle1 = plt.Circle((encoded_train[i,0], encoded_train[i,1]),noise_std, color='r', alpha = 0.2)
				ax.add_patch(circle1)
				#plt.plot(enc_orig[:,0]. enc_orig[:,1], "go")
			plt.plot(encoded_train[0,0], encoded_train[0,1], 'b*')
			plt.plot(encoded_train[1,0], encoded_train[1,1], 'b*')
			plt.scatter(encoded_train[:,0], encoded_train[:,1], color="gray", marker="o", s = 10, edgecolors="black", alpha=0.8)
			plt.axis("equal")
			#ax.add_patch(circle1)

			plt.savefig(results_directory + "/" + "dret.pdf")
			plt.close()

			plt.figure()
			plt.hist2d(encoded_train[:,0], encoded_train[:,1],100)
			plt.plot(encoded_train[0,0], encoded_train[0,1], 'b*')
			plt.plot(encoded_train[1,0], encoded_train[1,1], 'b*')
			plt.colorbar()
			plt.savefig(results_directory + "/" + "dret2.pdf")



		############################### gconc ###############################
		try:
			baseline_genotype_concordance = get_baseline_gc(true_genotypes)
		except:
			baseline_genotype_concordance = None
		plt.figure()
		outfilename = "{0}/genotype_concordances.csv".format(results_directory)
		epochs_combined, genotype_concs_combined = write_metric_per_epoch_to_csv(outfilename, genotype_concs_train, epochs)

		plt.plot(epochs_combined, genotype_concs_combined, label="train", c="orange")
		if baseline_genotype_concordance:
			plt.plot([epochs_combined[0], epochs_combined[-1]], [baseline_genotype_concordance, baseline_genotype_concordance], label="baseline", c="black")

		plt.xlabel("Epoch")
		plt.ylabel("Genotype concordance")

		plt.savefig(results_directory + "/" + "genotype_concordances.pdf")

		plt.close()

		import shutil
		#try: shutil.rmtree(weights_dir)
		#except: pass
		try: shutil.rmtree("{0}/{1}".format(train_directory, "weights_temp"))
		except: pass



	elif arguments["project"] and not isChief:
		print("Work has ended for this worker, now relying only on the Chief :)")
		exit(0)

	if (arguments['evaluate'] or arguments['animate'] or arguments['plot'])  :# and isChief:
		if os.path.isfile(encoded_data_file):
			encoded_data = h5py.File(encoded_data_file, 'r')
		else:
			chief_print("------------------------------------------------------------------------")
			chief_print("Error: File {0} not found.".format(encoded_data_file))
			chief_print("------------------------------------------------------------------------")
			exit(1)

		epochs = get_projected_epochs(encoded_data_file)

		if arguments['epoch']:
			epoch = int(arguments['epoch'])
			if epoch in epochs:
				epochs = [epoch]
			else:
				chief_print("------------------------------------------------------------------------")
				chief_print("Error: Epoch {0} not found in {1}.".format(epoch, encoded_data_file))
				chief_print("------------------------------------------------------------------------")
				exit(1)

		if doing_clustering:
			if arguments['animate']:
				chief_print("------------------------------------------------------------------------")
				chief_print("Error: Animate not supported for genetic clustering model.")
				chief_print("------------------------------------------------------------------------")
				exit(1)


			if arguments['plot'] and not superpopulations_file:
				chief_print("------------------------------------------------------------------------")
				chief_print("Error: Plotting of genetic clustering results requires a superpopulations file.")
				chief_print("------------------------------------------------------------------------")
				exit(1)


	if arguments['animate']:

		print("Animating epochs {}".format(epochs))

		FFMpegWriter = animation.writers['ffmpeg']
		scatter_points_per_epoch = []
		colors_per_epoch = []
		markers_per_epoch = []
		edgecolors_per_epoch = []

		ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")

		for epoch in epochs:
			print("########################### epoch {0} ###########################".format(epoch))

			encoded_train = read_h5(encoded_data_file, "{0}_encoded_train".format(epoch))

			coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)
			name = ""

			if superpopulations_file:
				scatter_points, colors, markers, edgecolors = \
					plot_coords_by_superpop(coords_by_pop, name, superpopulations_file, plot_legend=False, savefig=False)
				suffix = "_by_superpop"
			else:
				try:
					scatter_points, colors, markers, edgecolors = plot_coords_by_pop(coords_by_pop, name, savefig=False)
					suffix = "_by_pop"
				except:
					scatter_points, colors, markers, edgecolors = plot_coords(encoded_train, name, savefig=False)
					suffix = ""

			scatter_points_per_epoch.append(scatter_points)
			colors_per_epoch.append(colors)
			markers_per_epoch.append(markers)
			edgecolors_per_epoch.append(edgecolors)

		make_animation(epochs, scatter_points_per_epoch, colors_per_epoch, markers_per_epoch, edgecolors_per_epoch, "{0}/{1}{2}".format(results_directory, "dimred_animation", suffix))

	if arguments['evaluate']:

		print("Evaluating epochs {}".format(epochs))

		# all metrics assumed to have a single value per epoch
		metric_names = arguments['metrics'].split(",")
		metrics = dict()

		for m in metric_names:
			metrics[m] = []

		ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")
		pop_list = []

		for pop in ind_pop_list_train[:, 1]:
			try:
				pop_list.append(pop.decode("utf-8"))
			except:
				pass

		for epoch in epochs:
			print("########################### epoch {0} ###########################".format(epoch))

			encoded_train = read_h5(encoded_data_file, "{0}_encoded_train".format(epoch))

			coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

			### count how many f1 scores were doing
			f1_score_order = []
			num_f1_scores = 0
			for m in metric_names:
				if m.startswith("f1_score"):
					num_f1_scores += 1
					f1_score_order.append(m)

			f1_scores_by_pop = {}
			f1_scores_by_pop["order"] = f1_score_order

			for pop in coords_by_pop.keys():
				f1_scores_by_pop[pop] = ["-" for i in range(num_f1_scores)]
			f1_scores_by_pop["avg"] = ["-" for i in range(num_f1_scores)]

			for m in metric_names:

				if m == "hull_error":
					coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)
					n_latent_dim = encoded_train.shape[1]
					if n_latent_dim == 2:
						min_points_required = 3
					else:
						min_points_required = n_latent_dim + 2
					hull_error = convex_hull_error(coords_by_pop, plot=False, min_points_required= min_points_required)
					print("------ hull error : {}".format(hull_error))

					metrics[m].append(hull_error)

				elif m.startswith("f1_score"):
					this_f1_score_index = f1_score_order.index(m)

					k = int(m.split("_")[-1])
					# num_samples_required = np.ceil(k/2.0) + 1 + (k+1) % 2
					num_samples_required = 1

					pops_to_use = get_pops_with_k(num_samples_required, coords_by_pop)

					if len(pops_to_use) > 0 and "{0}_{1}".format(m, pops_to_use[0]) not in metrics.keys():
						for pop in pops_to_use:
							try:
								pop = pop.decode("utf-8")
							except:
								pass
							metric_name_this_pop = "{0}_{1}".format(m, pop)
							metrics[metric_name_this_pop] = []


					f1_score_avg, f1_score_per_pop = f1_score_kNN(encoded_train, pop_list, pops_to_use, k = k)
					print("------ f1 score with {0}NN :{1}".format(k, f1_score_avg))
					metrics[m].append(f1_score_avg)
					assert len(f1_score_per_pop) == len(pops_to_use)
					f1_scores_by_pop["avg"][this_f1_score_index] =  "{:.4f}".format(f1_score_avg)

					for p in range(len(pops_to_use)):
						try:
							pop = pops_to_use[p].decode("utf-8")
						except:
							pop = pops_to_use[p]

						metric_name_this_pop = "{0}_{1}".format(m, pop)
						metrics[metric_name_this_pop].append(f1_score_per_pop[p])
						f1_scores_by_pop[pops_to_use[p]][this_f1_score_index] =  "{:.4f}".format(f1_score_per_pop[p])

				else:
					print("------------------------------------------------------------------------")
					print("Error: Metric {0} is not implemented.".format(m))
					print("------------------------------------------------------------------------")

			write_f1_scores_to_csv(results_directory, "epoch_{0}".format(epoch), superpopulations_file, f1_scores_by_pop, coords_by_pop)

		for m in metric_names:

			plt.plot(epochs, metrics[m], label="train", c="orange")
			plt.xlabel("Epoch")
			plt.ylabel(m)
			plt.savefig("{0}/{1}.pdf".format(results_directory, m))
			plt.close()

			outfilename = "{0}/{1}.csv".format(results_directory, m)
			with open(outfilename, mode='w') as res_file:
				res_writer = csv.writer(res_file, delimiter=',')
				res_writer.writerow(epochs)
				res_writer.writerow(metrics[m])

	if arguments['plot']:

		print("Plotting epochs {}".format(epochs))

		ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")
		pop_list = []

		for pop in ind_pop_list_train[:, 1]:
			try:
				pop_list.append(pop.decode("utf-8"))
			except:
				pass

		for epoch in epochs:
			print("########################### epoch {0} ###########################".format(epoch))

			encoded_train = read_h5(encoded_data_file, "{0}_encoded_train".format(epoch))

			coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

			if superpopulations_file:

				coords_by_pop = get_coords_by_pop(data_prefix, encoded_train, ind_pop_list = ind_pop_list_train)

				if doing_clustering:
					plot_clusters_by_superpop(coords_by_pop, "{0}/clusters_e_{1}".format(results_directory, epoch), superpopulations_file, write_legend = epoch == epochs[0])
				else:
					scatter_points, colors, markers, edgecolors = \
						plot_coords_by_superpop(coords_by_pop, "{0}/dimred_e_{1}_by_superpop".format(results_directory, epoch), superpopulations_file, plot_legend = epoch == epochs[0])

			else:
				try:
					plot_coords_by_pop(coords_by_pop, "{0}/dimred_e_{1}_by_pop".format(results_directory, epoch))
				except:
					plot_coords(encoded_train, "{0}/dimred_e_{1}".format(results_directory, epoch))


if __name__ == "__main__":
	main()