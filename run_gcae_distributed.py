"""GenoCAE.

Usage:
  run_gcae_distributed.py train --datadir=<name> --data=<name> --model_id=<name> --train_opts_id=<name> --data_opts_id=<name> --save_interval=<num> --epochs=<num> [--resume_from=<num> --trainedmodeldir=<name> ]
  run_gcae_distributed.py project --datadir=<name>   [ --data=<name> --model_id=<name>  --train_opts_id=<name> --data_opts_id=<name> --superpops=<name> --epoch=<num> --trainedmodeldir=<name>   --pdata=<name> --trainedmodelname=<name>]
  run_gcae_distributed.py plot --datadir=<name> [  --data=<name>  --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name>  --pdata=<name> --trainedmodelname=<name>]
  run_gcae_distributed.py animate --datadir=<name>   [ --data=<name>   --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name> --pdata=<name> --trainedmodelname=<name>]
  run_gcae_distributed.py evaluate --datadir=<name> --metrics=<name>  [  --data=<name>  --model_id=<name> --train_opts_id=<name> --data_opts_id=<name>  --superpops=<name> --epoch=<num> --trainedmodeldir=<name>  --pdata=<name> --trainedmodelname=<name>]

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


"""

from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
from docopt import docopt, DocoptExit
import tensorflow as tf
from tensorflow.keras import Model, layers
from datetime import datetime
from utils.data_handler import  get_saved_epochs, get_projected_epochs, write_h5, read_h5, get_coords_by_pop, convex_hull_error, f1_score_kNN, plot_genotype_hist, to_genotypes_sigmoid_round, to_genotypes_invscale_round, GenotypeConcordance, get_pops_with_k, get_ind_pop_list_from_map, get_baseline_gc, write_metric_per_epoch_to_csv, alt_data_generator, parquet_converter
from utils.visualization import plot_coords_by_superpop, plot_clusters_by_superpop, plot_coords, plot_coords_by_pop, make_animation, write_f1_scores_to_csv
import utils.visualization
import utils.layers
import json
import numpy as np
import time
import os
import math
import matplotlib.pyplot as plt
import csv
import copy
import h5py
import matplotlib.animation as animation
from pathlib import Path
from utils.set_tf_config_berzelius import set_tf_config
from utils.data_handler import alt_data_generator



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

		chief_print("\n______________________________ Building model ______________________________")
		# variable that keeps track of the size of layers in encoder, to be used when constructing decoder.
		ns=[]
		ns.append(n_markers)

		first_layer_def = model_architecture["layers"][0]
		layer_module = getattr(eval(first_layer_def["module"]), first_layer_def["class"])
		layer_args = first_layer_def["args"]
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

		# add all layers except first
		for layer_def in model_architecture["layers"][1:]:
			layer_module = getattr(eval(layer_def["module"]), layer_def["class"])
			layer_args = layer_def["args"]

			for arg in ["size", "layers", "units", "shape", "target_shape", "output_shape", "kernel_size", "strides"]:

				if arg in layer_args.keys():
					layer_args[arg] = eval(str(layer_args[arg]))

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
			self.ms_variable = tf.Variable(random_uniform(shape = (1, n_markers), dtype=tf.float32), name="marker_spec_var")
			self.nms_variable = tf.Variable(random_uniform(shape = (1, n_markers), dtype=tf.float32), name="nmarker_spec_var")
		else:
			chief_print("No marker specific variable.")


	def call(self, input_data, is_training = True, verbose = False):
		'''
		The forward pass of the model. Given inputs, calculate the output of the model.

		:param input_data: input data
		:param is_training: if called during training
		:param verbose: chief_print the layers and their shapes
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
				x = layer_def(x)

			# If this is a clustering model then we add noise to the layer first in this step
			# and the next layer, which is sigmoid, is the actual encoding.
			if layer_name == "encoded_raw":
				have_encoded_raw = True
				if self.noise_std:
					x = self.noise_layer(x, training = is_training)
				encoded_data_raw = x

			# If this is the encoding layer, we add noise if we are training
			if layer_name == "encoded":
				if self.noise_std and not have_encoded_raw:
					x = self.noise_layer(x, training = is_training)
				encoded_data = x

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

		if self.regularizer:
			reg_module = eval(self.regularizer["module"])
			reg_name = getattr(reg_module, self.regularizer["class"])
			reg_func = reg_name(float(self.regularizer["reg_factor"]))

			# if this is a clustering model then the regularization is added to the raw encoding, not the softmaxed one
			if have_encoded_raw:
				reg_loss = reg_func(encoded_data_raw)
			else:
				reg_loss = reg_func(encoded_data)
			self.add_loss(reg_loss)

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

	:param y_pred: (n_samples x n_markers) tensor of raw network output for each sample and site
	:return: (n_samples x n_markers x 3 tensor) of genotype probabilities for each sample and site
	'''

	if len(y_pred.shape) == 2:
		alfreq = tf.keras.activations.sigmoid(y_pred)
		alfreq = tf.expand_dims(alfreq, -1)
		return tf.concat(((1-alfreq) ** 2, 2 * alfreq * (1 - alfreq), alfreq ** 2), axis=-1)
	else:
		return tf.nn.softmax(y_pred)

def save_ae_weights(epoch, train_directory, autoencoder):
	weights_file_prefix = train_directory + "/weights/" + str(epoch)
	startTime = datetime.now()
	if os.path.isdir(weights_file_prefix):
		newname = train_directory+"_"+str(time.time())
		os.rename(train_directory, newname)
		chief_print("... renamed " + train_directory + " to  " + newname)

	if isChief: autoencoder.save_weights(weights_file_prefix, save_format ="tf")
	save_time = (datetime.now() - startTime).total_seconds()
	chief_print("-------- Saving weights: {0} time: {1}".format(weights_file_prefix, save_time))


def save_ae_weights_multiworker(epoch, train_directory, autoencoder):

	if "isChief" in os.environ:
        
		if os.environ["isChief"] == "true":
			weights_file_prefix = train_directory + "/weights/" + str(epoch)
		else:
			weights_file_prefix = train_directory + "/weights_temp/"+str(os.environ["SLURMD_NODENAME"]) + str(epoch)
			

	else:
		print(str)
		weights_file_prefix = train_directory + "/weights/" + str(epoch)
	
	startTime = datetime.now()
	if os.path.isdir(weights_file_prefix):
		newname = train_directory+"_"+str(time.time())
		os.rename(train_directory, newname)
		chief_print("... renamed " + train_directory + " to  " + newname)

	autoencoder.save_weights(weights_file_prefix, save_format ="tf")

	save_time = (datetime.now() - startTime).total_seconds()
	chief_print("-------- Saving weights: {0} time: {1}".format(weights_file_prefix, save_time))





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

	
	 

if __name__ == "__main__":
	timer = time.perf_counter()
	chief_print("tensorflow version {0}".format(tf.__version__))
	tf.keras.backend.set_floatx('float32')
	tf.random.set_seed(2)
	np.random.seed(2)


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
		print(num_workers)
		if num_workers > 1 and arguments["train"]:
			#Here, NCCL is what I would want to use - it is Nvidias own implementation of reducing over the devices. However, this induces a segmentation fault, and the default value works.
			# had some issues, I think this was resolved by updating to TensorFlow 2.7 from 2.5. 
			# However, the image is built on the cuda environment for 2.5 This leads to problems when profiling


			strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=tf.distribute.cluster_resolver.TFConfigClusterResolver(),
																communication_options=tf.distribute.experimental.CommunicationOptions(
																implementation=tf.distribute.experimental.CollectiveCommunication.NCCL) 
																)

			num_physical_gpus = len(tf.config.list_physical_devices(device_type='GPU'))
			
			print(tf.config.list_physical_devices(device_type='GPU'))
			gpus = ["gpu:"+ str(i) for i in range(num_physical_gpus)]
			print(gpus)
			
		#elif num_workers > 1 and (arguments["evaluate"] or arguments["project"]):
		else:
			if not isChief:
				print("Work has ended for this worker, now relying only on the Chief :)")
				exit(0)	

			num_physical_gpus = len(tf.config.list_physical_devices(device_type='GPU'))
			
			print(tf.config.list_physical_devices(device_type='GPU'))
			gpus = ["gpu:"+ str(i) for i in range(num_physical_gpus)]
			print(gpus)
			strategy =  tf.distribute.MirroredStrategy(devices = gpus, cross_device_ops=tf.distribute.NcclAllReduce())
			
			slurm_job = 0
		os.environ["isChief"] = json.dumps((isChief))

	else:
		
		slurm_job = 0
		strategy =  tf.distribute.MirroredStrategy()
	
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

		data_opts_id = trainedmodelname.split(".")[3]
		train_opts_id = trainedmodelname.split(".")[2]
		model_id = trainedmodelname.split(".")[1]
		data = trainedmodelname.split(".")[4]

	else:
		data = arguments['data']
		data_opts_id = arguments["data_opts_id"]
		train_opts_id = arguments["train_opts_id"]
		model_id = arguments["model_id"]
		train_directory = False

	with open("{}/data_opts/{}.json".format(GCAE_DIR, data_opts_id)) as data_opts_def_file:
		data_opts = json.load(data_opts_def_file)

	with open("{}/train_opts/{}.json".format(GCAE_DIR, train_opts_id)) as train_opts_def_file:
		train_opts = json.load(train_opts_def_file)

	with open("{}/models/{}.json".format(GCAE_DIR, model_id)) as model_def_file:
		model_architecture = json.load(model_def_file)

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


	batch_size = train_opts["batch_size"]# * num_devices
	

	learning_rate = train_opts["learning_rate"]
	regularizer = train_opts["regularizer"]

	superpopulations_file = arguments['superpops']
	if superpopulations_file and not os.path.isabs(os.path.dirname(superpopulations_file)):
		superpopulations_file="{}/{}/{}".format(GCAE_DIR, os.path.dirname(superpopulations_file), Path(superpopulations_file).name)

	norm_opts = data_opts["norm_opts"]
	norm_mode = data_opts["norm_mode"]
	validation_split = data_opts["validation_split"]

	if "sparsifies" in data_opts.keys():
		sparsify_input = True
		missing_mask_input = True
		n_input_channels = 2
		sparsifies = data_opts["sparsifies"]

	else:
		sparsify_input = False
		missing_mask_input = False
		n_input_channels = 1
		sparsifies = False

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
		train_directory = trainedmodeldir + "ae." + model_id + "." + train_opts_id + "." + data_opts_id  + "." + data

	if arguments["pdata"]:
		pdata = arguments["pdata"]
	else:
		pdata = data

	data_prefix = datadir + pdata
	results_directory = "{0}/{1}".format(train_directory, pdata)

	if not os.path.exists(train_directory):
		if isChief:
			#os.mkdir(trainedmodeldir)
			os.mkdir(train_directory)
	#if not os.path.exists(results_directory):
		#if isChief:
			os.mkdir(results_directory)
		
	
	encoded_data_file = "{0}/{1}/{2}".format(train_directory, pdata, "encoded_data.h5")

	if "noise_std" in train_opts.keys():
		noise_std = train_opts["noise_std"]
	else:
		noise_std = False
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


	if (arguments['train'] or arguments['project']):

		# Create parquet file from the plink files if it does not already exist, and create an instance of the dataset. 
		# The number of SNP markers are needed in the loss function definitions
		max_mem_size = 5 * 10**10 # this is approx 50GB
		filebase = data_prefix
		t3 = time.perf_counter()
		parquet_converter(filebase, max_mem_size=max_mem_size)
		print("time for creating parquet: " + str(time.perf_counter()- t3))
		data = alt_data_generator(filebase= data_prefix, 
						batch_size = batch_size,
						normalization_mode = norm_mode,
						normalization_options = norm_opts,
						impute_missing = fill_missing)

		n_markers = data.n_markers
		

		loss_def = train_opts["loss"]
		loss_class = getattr(eval(loss_def["module"]), loss_def["class"])
		if "args" in loss_def.keys():
			loss_args = loss_def["args"]
		else:
			loss_args = dict()
		loss_obj = loss_class(**loss_args)

		with strategy.scope():

			if loss_class == tf.keras.losses.CategoricalCrossentropy or loss_class == tf.keras.losses.KLDivergence:
				
				def loss_func(y_pred, y_true):
					y_pred = y_pred[:, 0:n_markers]

					if not fill_missing:
						orig_nonmissing_mask = get_originally_nonmissing_mask(y_true)

					y_pred = alfreqvector(y_pred)
					y_true = tf.one_hot(tf.cast(y_true * 2, tf.uint8), 3)*0.9997 + 0.0001
					
					if not fill_missing:
						y_pred = y_pred[orig_nonmissing_mask]
						y_true = y_true[orig_nonmissing_mask]

					if loss_class ==	tf.keras.losses.CategoricalCrossentropy:
						loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

					if loss_class == loss_class == tf.keras.losses.KLDivergence:
						loss_obj = tf.keras.losses.KLDivergence(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

					per_example_loss = loss_obj(y_true, y_pred)
					per_example_loss /= n_markers
					loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size
														) 

					return loss
				def loss_func_non_dist(y_pred, y_true):
					y_pred = y_pred[:, 0:n_markers]

					if not fill_missing:
						orig_nonmissing_mask = get_originally_nonmissing_mask(y_true)

					y_pred = alfreqvector(y_pred)
					y_true = tf.one_hot(tf.cast(y_true * 2, tf.uint8), 3)*0.9997 + 0.0001

					if not fill_missing:
						y_pred = y_pred[orig_nonmissing_mask]
						y_true = y_true[orig_nonmissing_mask]

					return loss_obj(y_pred = y_pred, y_true = y_true)

			else:

				def loss_func(y_pred, y_true):
					y_pred = y_pred[:, 0:n_markers]

					if not fill_missing:
						orig_nonmissing_mask = get_originally_nonmissing_mask(y_true)
						y_pred = y_pred[orig_nonmissing_mask]
						y_true = y_true[orig_nonmissing_mask]

					if loss_class == tf.keras.losses.MeanSquaredError:
						loss_obj = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE) 
					per_example_loss = loss_obj(y_true = y_true, y_pred = y_pred)			

					loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size= batch_size) 
					return loss
						
			@tf.function
			def train_step(model, optimizer, loss_function, input, targets):
				
				with tf.GradientTape() as tape:
					output, _ = model(input)
					loss_value = loss_function(y_pred = output, y_true = targets)
					loss_value += tf.nn.scale_regularization_loss(sum(model.losses))	
					
				grads = tape.gradient(loss_value, model.trainable_variables)
				optimizer.apply_gradients(zip(grads, model.trainable_variables),experimental_aggregate_gradients=False )

				return loss_value  


			@tf.function
			def distributed_train_step(model, optimizer, loss_function, input, targets):

				per_replica_losses = strategy.run(train_step, args=(model, optimizer, loss_function, input, targets))
				loss = strategy.reduce("SUM", per_replica_losses, axis=None)

				return loss 

			@tf.function
			def compute_loss(model, loss_function, input, targets):
					output, _ = model(input, is_training = False)
					loss_value = loss_function(y_pred = output, y_true = targets)
					
					loss_value += tf.nn.scale_regularization_loss(sum(model.losses))	
					return loss_value

			@tf.function
			def dist_compute_loss(model, loss_function, input, targets):
			
				per_replica_losses = strategy.run(compute_loss, args=(model, loss_function, input, targets))
				loss = strategy.reduce("SUM", per_replica_losses, axis=None)
				return loss


		try:
			resume_from = int(arguments["resume_from"])
			if resume_from < 1:
				saved_epochs = get_saved_epochs(train_directory)
				resume_from = saved_epochs[-1]
		except:
			resume_from = False


	if arguments['train']:
		epochs = int(arguments["epochs"])
		save_interval = int(arguments["save_interval"])




		data = alt_data_generator(filebase= data_prefix, 
						batch_size = batch_size,
						normalization_mode = norm_mode,
						normalization_options = norm_opts,
						impute_missing = fill_missing,
						sparsifies  = sparsifies)

		n_markers = data.n_markers
		data.define_validation_set2(validation_split= 0.2)

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
			optimizer = tf.optimizers.Adam(learning_rate = lr_schedule)


			#batch_dist_input, batch_dist_target, _ = next(ds.as_numpy_iterator())
			#output_test, encoded_alt_data_generator = autoencoder(batch_dist_input, is_training = False, verbose = True)

			if resume_from:
				chief_print("\n______________________________ Resuming training from epoch {0} ______________________________".format(resume_from))
				weights_file_prefix = "{0}/{1}/{2}".format(train_directory, "weights", resume_from)
				chief_print("Reading weights from {0}".format(weights_file_prefix))
				# get a single BATCH to run through optimization to reload weights and optimizer variables
				# This initializes the variables used by the optimizers,
				# as well as any stateful metric variables
				distributed_train_step(autoencoder, optimizer, loss_func, batch_dist_input, batch_dist_target)
				autoencoder.load_weights(weights_file_prefix)



		
		######### Create objects for tensorboard summary ###############################
		train_writer = tf.summary.create_file_writer(train_directory + '/train')
		valid_writer = tf.summary.create_file_writer(train_directory + '/valid')
		######################################################

		# train losses per epoch
		losses_t = []
		# valid losses per epoch
		losses_v = []

		""""
		Enable the below command to log the runs in the profiler, also de-comment the line after the training loop. 

		 Used for debugging the code / profiling different runs. Visualized in TensorBoard. I like to use ThinLinc to look at it, seems like the
		 least effort solution to get tensorboard up and running. Possible to do it in other ways, but more painful
		"""
		suffix = ""
		if slurm_job:
			suffix = str(os.environ["SLURMD_NODENAME"])
		logs = train_directory+ "/logdir/"  + datetime.now().strftime("%Y%m%d-%H%M%S") + suffix
		
		profile = 0 

		for e in range(1,epochs+1):
				if e ==2:

					if profile : tf.profiler.experimental.start(logs)

				train_losses = []
				startTime = datetime.now()
				effective_epoch = e + resume_from
				losses_t_batches = []
				losses_v_batches = []
				train_batch_loss = 0

				t0 = time.perf_counter()
				for batch_dist_input, batch_dist_target, _  in dds:
					train_batch_loss += distributed_train_step(autoencoder, optimizer, loss_func, batch_dist_input, batch_dist_target)
			
				train_loss_this_epoch = train_batch_loss /n_train_samples  
		
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

				chief_print("")
				chief_print("Epoch: {}/{}...".format(effective_epoch, epochs+resume_from))
				chief_print("--- Train loss: {:.4f} time: {}".format(train_loss_this_epoch,  train_time))

				if n_valid_samples > 0:

					startTime = datetime.now()
					valid_loss_batch = 0

					for input_valid_batch, targets_valid_batch, _  in dds_validation:
						valid_loss_batch += dist_compute_loss(autoencoder, loss_func, input_valid_batch, targets_valid_batch)

					valid_loss_this_epoch = valid_loss_batch /n_valid_samples 

					with valid_writer.as_default():
						tf.summary.scalar('loss', valid_loss_this_epoch, step=step_counter)

					losses_v.append(valid_loss_this_epoch)
					valid_time = (datetime.now() - startTime).total_seconds()
					chief_print("--- Valid loss: {:.4f}  time: {}".format(valid_loss_this_epoch, valid_time))

				if e % save_interval == 0:
					save_ae_weights_multiworker(effective_epoch, train_directory, autoencoder)

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
				plt.axvline(min_valid_loss_epoch, color="black")
				plt.text(min_valid_loss_epoch + 0.1, 0.5,'min valid loss at epoch {}'.format(int(min_valid_loss_epoch)),
						rotation=90,
						transform=ax.get_xaxis_text1_transform(0)[0])

			plt.xlabel("Epoch")
			plt.ylabel("Loss function value")
			plt.legend()
			plt.savefig("{}/losses_from_train.pdf".format(train_directory))
			plt.close()

			chief_print("Done training. Wrote to {0}".format(train_directory))
			chief_print("TOTAL TRAINING TIME : " + str( time.perf_counter() - timer))


	
	if arguments['project']: 
		

		try:
			if autoencoder in locals:
				del autoencoder
		except: 
			pass
		try:
			if data in locals:
				del data
		except: 
			pass
		
		projected_epochs = get_projected_epochs(encoded_data_file)
		
		if arguments['epoch']:
			epoch = int(arguments['epoch'])
			epochs = [epoch]

		else:
			epochs = get_saved_epochs(train_directory)

		for projected_epoch in projected_epochs:
			try:
				epochs.remove(projected_epoch)
			except:
				continue

		chief_print("Projecting epochs: {0}".format(epochs))
		chief_print("Already projected: {0}".format(projected_epochs))

		# Make this larger. For some reason a low project batch size resulted in division by zero in the normalization, which yields a bad scaler
		batch_size_project = batch_size 
		sparsify_fraction = 0.0

	

		data.sparsifies = [sparsify_fraction]

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
		print("n_train_samples: " + str(n_train_samples))

		n_train_batches, n_train_samples_last_batch = get_batches(n_train_samples, batch_size_project)
		n_valid_samples = 0
		print("n_train_samples_last_batch: " + str(n_train_samples_last_batch))
		batch_size_valid = 1
		n_valid_batches, n_valid_samples_last_batch = get_batches(n_valid_samples, batch_size_valid)

		data.n_valid_samples_last_batch = n_valid_samples_last_batch
		data.n_train_samples_last_batch = n_train_samples_last_batch 

		data.missing_mask_input = missing_mask_input
		
		#####################
		
		# loss function of the train set per epoch
		losses_train = []

		# genotype concordance of the train set per epoch
		genotype_concs_train = []
		genotype_concordance_metric = GenotypeConcordance()

		autoencoder = Autoencoder(model_architecture, n_markers, noise_std, regularizer)

		scatter_points_per_epoch = []
		colors_per_epoch = []
		markers_per_epoch = []
		edgecolors_per_epoch = []

		data.batch_size = batch_size_project 
		chunk_size = 5  * data.batch_size
		
		# HERE WE NEED TO "NOT SHUFFLE" THE DATASET, IN ORDER TO GET EVALUATE TO WORK AS INTENDED (otherwise, there is a problem with the ordering, works on its own, 
		# but works differently when directly compared to original implementation) 
		ds = data.create_dataset(chunk_size, "training", shuffle = False)




		for epoch in epochs:
			chief_print("########################### epoch {0} ###########################".format(epoch))
			weights_file_prefix = "{0}/{1}/{2}".format(train_directory, "weights", epoch)
			weights_dir = "{0}/{1}".format(train_directory, "weights")
			chief_print("Reading weights from {0}".format(weights_file_prefix))

			autoencoder.load_weights(weights_file_prefix)


			ind_pop_list_train = np.empty((0,2))
			encoded_train = np.empty((0, n_latent_dim))
			decoded_train = None
			targets_train = np.empty((0, n_markers))

			loss_value_per_train_batch = []
			genotype_conc_per_train_batch = []
			loss_train_batch = 0

			for input_train_batch, targets_train_batch, ind_pop_list_train_batch in ds:

				decoded_train_batch, encoded_train_batch  = autoencoder(input_train_batch, is_training = False)
				loss_train_batch = loss_func_non_dist(y_pred = decoded_train_batch, y_true = targets_train_batch)
				loss_train_batch += sum((autoencoder.losses))

				encoded_train = np.concatenate((encoded_train, encoded_train_batch), axis=0)

				if decoded_train is None:
					decoded_train = np.copy(decoded_train_batch[:,0:n_markers])
				else:
					decoded_train = np.concatenate((decoded_train, decoded_train_batch[:,0:n_markers]), axis=0)

				ind_pop_list_train = np.concatenate((ind_pop_list_train, np.array(ind_pop_list_train_batch,dtype = "U21")), axis=0)
				
				targets_train = np.concatenate((targets_train, targets_train_batch[:,0:n_markers]), axis=0)
				loss_value_per_train_batch.append(loss_train_batch)

			ind_pop_list_train = np.array(ind_pop_list_train)
			encoded_train = np.array(encoded_train)

			
			loss_value = np.sum(loss_value_per_train_batch)  / n_train_samples  

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


		
			write_h5(encoded_data_file, "{0}_encoded_train".format(epoch), encoded_train)


	
		try:
			plot_genotype_hist(np.array(genotypes_output), "{0}/{1}_e{2}".format(results_directory, "output_as_genotypes", epoch))
			plot_genotype_hist(np.array(true_genotypes), "{0}/{1}".format(results_directory, "true_genotypes"))
		except:
			pass

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

		############################### gconc ###############################
		try:
			baseline_genotype_concordance = get_baseline_gc(true_genotypes)
		except:
			baseline_genotype_concordance = None

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
		try: shutil.rmtree(weights_dir)
		except: pass
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

		chief_print("Animating epochs {}".format(epochs))
		FFMpegWriter = animation.writers['ffmpeg']
		scatter_points_per_epoch = []
		colors_per_epoch = []
		markers_per_epoch = []
		edgecolors_per_epoch = []

		ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")

		for epoch in epochs:
			chief_print("########################### epoch {0} ###########################".format(epoch))
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

	if arguments['evaluate'] :# and isChief: 

		chief_print("Evaluating epochs {}".format(epochs))

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
			chief_print("########################### epoch {0} ###########################".format(epoch))

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
					chief_print("------ hull error : {}".format(hull_error))

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
					chief_print("------ f1 score with {0}NN :{1}".format(k, f1_score_avg))
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
					chief_print("------------------------------------------------------------------------")
					chief_print("Error: Metric {0} is not implemented.".format(m))
					chief_print("------------------------------------------------------------------------")

			write_f1_scores_to_csv(results_directory, "epoch_{0}".format(epoch), superpopulations_file, f1_scores_by_pop, coords_by_pop)

		for m in metric_names:

			plt.plot(epochs, metrics[m], label="train", c="orange")
			plt.xlabel("Epoch")
			plt.ylabel(m)

			if isChief: plt.savefig("{0}/{1}.pdf".format(results_directory, m))
			plt.close()

			outfilename = "{0}/{1}.csv".format(results_directory, m)
			with open(outfilename, mode='w') as res_file:
				res_writer = csv.writer(res_file, delimiter=',')
				res_writer.writerow(epochs)
				res_writer.writerow(metrics[m])
	elif arguments['evaluate']  and not isChief: 
		print("Work has ended for this worker, now relying only on the Chief :)")
		exit(1)	
	if arguments['plot']:

		chief_print("Plotting epochs {}".format(epochs))

		ind_pop_list_train = read_h5(encoded_data_file, "ind_pop_list_train")
		pop_list = []

		for pop in ind_pop_list_train[:, 1]:
			try:
				pop_list.append(pop.decode("utf-8"))
			except:
				pass

		for epoch in epochs:
			chief_print("########################### epoch {0} ###########################".format(epoch))

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


	chief_print("TOTAL TIME : " + str( time.perf_counter() - timer))

	# This removes the saved weights. They take up waaaay too much space on disk for many large models / long runs.
	# After having projected/evaluated, we really dont need them anymore, unless we want to continue training after a checkpoint.
	# Not sure if that is something I will want to do in the near future however.
