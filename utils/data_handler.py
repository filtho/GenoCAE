import os
import numpy as np
import h5py
import utils.normalization as normalization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from scipy.spatial import ConvexHull
import functools
from scipy.spatial import Delaunay
import traceback
import sys
import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from scipy import stats
from pandas_plink import read_plink
import csv
import pyarrow.parquet as pq
import pandas as pd
from sklearn.preprocessing import  StandardScaler
from pysnptools.snpreader import Bed
import pyarrow as pa
import shutil

class data_generator_ae:
	'''
	Class to generate data for training and evaluation.

	If get_data is False then only ind pop list will be generated

	'''


	def __init__(self,
				 filebase,
				 normalization_mode = "smartPCAstyle",
				 normalization_options= {"flip":False, "missing_val":0.0},
				 get_genotype_data=True,
				 impute_missing = True):
		'''

		:param filebase: path + filename prefix of eigenstrat (eigenstratgeno/snp/fam) or plink (bed/bim/fam) data
		:param normalization_mode: how to normalize the genotypes. corresponds to functions in utils/normalization
		:param normalization_options: options for normalization
		:param get_genotype_data: whether ot not to compute and return genotypes (otherwise just metadata about number of samples, etc. is generated. faster)
		:param impute_missing: if true, genotypes that are missing in the original data are set to the most frequent genotype per marker.

		'''

		self.filebase = filebase
		self.missing_val = normalization_options["missing_val"]
		self.normalization_mode = normalization_mode
		self.normalization_options = normalization_options
		self.train_batch_location = 0
		self.impute_missing = impute_missing

		self._define_samples()

		if get_genotype_data:
			self._normalize()

	def _impute_missing(self, genotypes):
		'''
		Replace missing values in genotypes with the most frequent value per SNP.

		If two or more values occur the same number of times, take the smallest of them.
		NOTE: this means that the way genotypes are represented (num ALT or num REF alles) can affect
		which genotype gets imputed for those cases where HET occurs as many times as a HOM genotype.

		:param genotypes: (n_markers x n_samples) numpy array of genotypes, missing values represented by 9.
		'''

		for m in genotypes:
			m[m == 9.0] = get_most_frequent(m)


	def _define_samples(self):
		ind_pop_list = get_ind_pop_list(self.filebase)

		self.sample_idx_all = np.arange(len(ind_pop_list))
		self.sample_idx_train = np.arange(len(ind_pop_list))


		self.n_train_samples_orig = len(self.sample_idx_all)
		self.n_train_samples = self.n_train_samples_orig
		self.ind_pop_list_train_orig = ind_pop_list[self.sample_idx_all]
		self.train_set_indices = np.arange(self.n_train_samples)

		self.n_valid_samples = 0

	def _sparsify(self, mask, keep_fraction):
		'''
		Sparsify a mask defining data that is missing / present.

		0 represents missing
		1 represents present

		:param mask: int array (n x m)
		:param keep_fraction: probability to keep data
		'''
		mask[np.random.random_sample(mask.shape) > keep_fraction] = 0

	def _normalize(self):
		'''
		Normalize the genotype data.

		'''

		ind_pop_list = get_ind_pop_list(self.filebase)
		n_samples = len(ind_pop_list)

		try:
			genotypes = np.genfromtxt(self.filebase + ".eigenstratgeno", delimiter = np.repeat(1, n_samples))
			self.n_markers = len(genotypes)
		except:
			(genotypes, self.n_markers) = genfromplink(self.filebase)

		if self.impute_missing:
			self._impute_missing(genotypes)

		genotypes = np.array(genotypes, order='F') # Cheeky, this will then be transposed, so we have individual-major order

		genotypes_train = genotypes[:, self.sample_idx_all]

		normalization_method = getattr(normalization, "normalize_genos_"+self.normalization_mode)


		genotypes_train_normed, _, scaler = normalization_method(genotypes_train, np.array([]),
																					 get_scaler = True,
																					 **self.normalization_options)
		self.scaler = scaler
		
		self.genotypes_train_orig = np.array(genotypes_train_normed, dtype = np.dtype('f4'), order='C')

	def get_nonnormalized_data(self):
		'''
		Get the non-nornalized training data.
		Missing data represented by missing_val.

		:return: train data (n_samples x n_markers)

		'''
		ind_pop_list = get_ind_pop_list(self.filebase)
		n_samples = len(ind_pop_list)

		try:
			genotypes = np.genfromtxt(self.filebase + ".eigenstratgeno", delimiter = np.repeat(1, n_samples))
			self.n_markers = len(genotypes)
		except:
			(genotypes, self.n_markers) = genfromplink(self.filebase)

		if self.impute_missing:
			self._impute_missing(genotypes)
		else:
			genotypes[genotypes == 9.0] = self.missing_val

		genotypes_train = np.array(genotypes[:, self.sample_idx_train].T, order='C')

		return genotypes_train

	def define_validation_set(self, validation_split):
		'''
		Define a set of validation samples from original train samples.
		Stratified by population.

		Re-defines self.sample_idx_train and self.sample_idx_valid

		:param validation_split: proportion of samples to reserve for validation set

		If validation_split is less samples than there are populations, one sample per population is returned.
		'''


		# reset index of fetching train batch samples
		self.train_batch_location = 0

		_, _, self.sample_idx_train, self.sample_idx_valid = get_test_samples_stratified(self.genotypes_train_orig, self.ind_pop_list_train_orig, validation_split)

		self.sample_idx_train = np.array(self.sample_idx_train)
		self.sample_idx_valid = np.array(self.sample_idx_valid)

		self.train_set_indices = np.array(range(len(self.sample_idx_train)))
		self.n_valid_samples = len(self.sample_idx_valid)
		self.n_train_samples = len(self.sample_idx_train)


	def define_validation_set2(self, validation_split):
		'''
		Define a set of validation samples from original train samples.
		Stratified by population.

		Re-defines self.sample_idx_train and self.sample_idx_valid

		:param validation_split: proportion of samples to reserve for validation set

		If validation_split is less samples than there are populations, one sample per population is returned.
		'''


		# reset index of fetching train batch samples
		self.train_batch_location = 0

		self.sample_idx_train, self.sample_idx_valid = get_test_samples_stratified2(self.ind_pop_list_train_orig, validation_split)

		self.sample_idx_train = np.array(self.sample_idx_train)
		self.sample_idx_valid = np.array(self.sample_idx_valid)

		self.train_set_indices = np.array(range(len(self.sample_idx_train)))
		self.n_valid_samples = len(self.sample_idx_valid)
		self.n_train_samples = len(self.sample_idx_train)


	def get_valid_set(self, sparsify):
		'''

		:param sparsify:
		:return: input_data_valid (n_valid_samples x n_markers x 2): sparsified validation genotypes with mask specifying missing values.
																			   originally missing + removed by sparsify are indicated by value 0
				 target_data_valid (n_valid_samples x n_markers): original validation genotypes
				 ind_pop_list valid (n_valid_samples x 2) : individual and population IDs of validation samples

				 or
				 empty arrays if no valid set defined

		'''

		if self.n_valid_samples == 0:
			return np.array([]), np.array([]), np.array([])

		# n_valid_samples x n_markers x 2
		input_data_valid = np.full((len(self.sample_idx_valid),self.genotypes_train_orig.shape[1],2),1.0)

		genotypes_valid = np.copy(self.genotypes_train_orig[self.sample_idx_valid])

		mask_valid = np.full(input_data_valid[:,:,0].shape, 1)

		if not self.impute_missing:
			# set the ones that are originally missing to 0 in mask (sinc we havent imputed them with RR)
			mask_valid[np.where(genotypes_valid == self.missing_val)] = 0


		if sparsify > 0.0:

			self._sparsify(mask_valid, 1.0 - sparsify)

			missing_idx_valid = np.where(mask_valid == 0)

			genotypes_valid[missing_idx_valid] = self.missing_val

		input_data_valid[:,:,0] = genotypes_valid
		input_data_valid[:,:,1] = mask_valid

		targets = self.genotypes_train_orig[self.sample_idx_valid]

		return input_data_valid, targets, self.ind_pop_list_train_orig[self.sample_idx_valid]


	def reset_batch_index(self):
		'''
		Reset the internal sample counter for batches. So the next batch will start with sample 0.

		'''
		self.train_batch_location = 0

	def shuffle_train_samples(self):
		'''
		Shuffle the order of the train samples that batches are taken from
			'''

		p = np.random.permutation(len(self.sample_idx_train))
		self.train_set_indices = self.train_set_indices[p]

	def _get_indices_looped(self, n_samples):
		'''
		Get indices of n_samples from the train set. Start over at 0 if reaching the end.
		:param n_samples: number of samples to get
		:return: indices
		'''

		if self.train_batch_location + n_samples < len(self.sample_idx_train):
			idx = list(range(self.train_batch_location, self.train_batch_location + n_samples, 1))
			self.train_batch_location = (self.train_batch_location + n_samples) % len(self.sample_idx_train)
		else:
			idx = list(range(self.train_batch_location, len(self.sample_idx_train), 1)) + list(range(0, n_samples - (len(self.sample_idx_train)-self.train_batch_location) , 1))
			self.train_batch_location = n_samples - (len(self.sample_idx_train)-self.train_batch_location)


		return self.train_set_indices[idx]

	def get_train_batch(self, sparsify, n_samples_batch):
		'''
		Get n_samples_batch train samples, with genotypes randomly set to missing with probability sparsify.
		Fetch n_samples sequentially starting at index self.train_batch_location, looping over the current train set

		If validation set has been defined, return train samples exluding the validation samples.

		:param sparsify:
		:param n_samples_batch: number of samples in batch
		:return: input_data_train_batch (n_samples x n_markers x 2): sparsified  genotypes with mask specifying missing values of train batch.
																			   originally missing + removed by sparsify are indicated by value 0
				 target_data_train_batch (n_samples x n_markers): original genotypes of this train batch
				 ind_pop_list_train_batch (n_samples x 2) : individual and population IDs of train batch samples

		'''
		input_data_train = np.full((n_samples_batch, self.genotypes_train_orig.shape[1], 2), 1.0, dtype=np.dtype('f4'))

		indices_this_batch = self._get_indices_looped(n_samples_batch)
		genotypes_train = np.copy(self.genotypes_train_orig[self.sample_idx_train[indices_this_batch]])

		mask_train = np.full(input_data_train[:,:,0].shape, 1)

		if not self.impute_missing:
			# set the ones that are originally missing to 0 in mask (sinc we havent imputed them with RR)
			mask_train[np.where(genotypes_train == self.missing_val)] = 0

		if sparsify > 0.0:

			self._sparsify(mask_train, 1.0 - sparsify)

			# indices of originally missing data + artifically sparsified data
			missing_idx_train = np.where(mask_train == 0)

			# fill genotypes with original valid genotypes and sparsify according to binary_mask_train
			genotypes_train[missing_idx_train] = self.missing_val

		input_data_train[:,:,0] = genotypes_train
		input_data_train[:,:,1] = mask_train

		targets = self.genotypes_train_orig[self.sample_idx_train[indices_this_batch]]

		return input_data_train, targets, self.ind_pop_list_train_orig[self.sample_idx_train[indices_this_batch]]


	def get_train_set(self, sparsify):
		'''
		Get all train samples, with genotypes randomly set to missing with probability sparsify.
		Excluding validation samples.

		If validation set has been defined, return train samples exluding the validation samples.

		:param sparsify: fraction of data to remove
		:return: input_data_train (n_train_samples x n_markers x 2): sparsified  genotypes with mask specifying missing values of all train samples.
																			   originally missing + removed by sparsify are indicated by value 0
				 target_data_train (n_train_samples x n_markers): original genotypes of train samples
				 ind_pop_list_train (n_train_samples x 2) : individual and population IDs of train  samples

		'''
		# n_train_samples x n_markers x 2
		input_data_train = np.full((self.n_train_samples, self.genotypes_train_orig[self.sample_idx_train].shape[1], 2), 1.0)

		genotypes_train = np.copy(self.genotypes_train_orig[self.sample_idx_train])

		# the mask is all ones
		mask_train = np.full(input_data_train[:,:,0].shape, 1)

		if not self.impute_missing:
			# set the ones that are originally missing to 0 in mask (sinc we havent imputed them with RR)
			mask_train[np.where(genotypes_train == self.missing_val)] = 0


		if sparsify > 0.0:

			self._sparsify(mask_train, 1.0 - sparsify)

			# indices of originally missing data + artifically sparsified data
			missing_idx_train = np.where(mask_train == 0)

			# fill genotypes with original valid genotypes and sparsify according to binary_mask_train
			genotypes_train[missing_idx_train] = self.missing_val

		input_data_train[:,:,0] = genotypes_train
		input_data_train[:,:,1] = mask_train

		print("In DG.get_train_set: number of "+str(self.missing_val)+" genotypes in train: " + str( len(np.where(input_data_train[:,:,0] == self.missing_val)[0])))
		print("In DG.get_train_set: number of -9 genotypes in train: " + str( len(np.where(input_data_train[:,:,0] == -9)[0])))
		print("In DG.get_train_set: number of 0 values in train mask: " + str( len(np.where(input_data_train[:,:,1] == 0)[0])))
		targets = self.genotypes_train_orig[self.sample_idx_train]

		return input_data_train, targets, self.ind_pop_list_train_orig[self.sample_idx_train]


def in_hull(p, hull):
	"""
	from https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
	Test if points in `p` are in `hull`

	`p` should be a `NxK` coordinates of `N` points in `K` dimensions
	`hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
	coordinates of `M` points in `K`dimensions for which Delaunay triangulation
	will be computed
	"""

	if not isinstance(hull,Delaunay):
		hull = Delaunay(hull)

	return hull.find_simplex(p, bruteforce=True)>=0

def convex_hull_error(coords_by_pop, plot = False, min_points_required = 3):
	'''

	Calculate the hull error of projected coordinates of the populations defined by coords_by_pop

	For every population: calculate the fraction that other population's points make up of all the points inside THIS population's convex hull
	Return the average of this over populations.

	:param coords_by_pop: dict mapping population to list of coordinates (n_samples x n_dim)
	:param min_points_required: min number of points needed to define a simplex for the convex hull.

	:return: the hull error
	'''
	all_pop_coords = functools.reduce(lambda a,b : a+b, [coords_by_pop[pop] for pop in coords_by_pop.keys()])

	num_pops = len(coords_by_pop.keys())
	frac_sum = 0.0

	try:
		num_pops = len(list(coords_by_pop.keys()))
		for pop in coords_by_pop.keys():
			other_pops = list(coords_by_pop.keys())
			assert len(other_pops) == num_pops
			other_pops.remove(pop)
			this_pop_coords = np.array(coords_by_pop[pop])
			num_points_in_pop = float(len(this_pop_coords))

			other_pop_coords = np.concatenate([np.array(coords_by_pop[popu]) for popu in other_pops])

			assert  len(other_pop_coords) + num_points_in_pop == len(all_pop_coords), "Number of inds does not match up!"

			if num_points_in_pop >= min_points_required:
				num_other_points_in_hull = sum(in_hull(other_pop_coords, this_pop_coords))
				frac_other_pop_points_in_hull = num_other_points_in_hull / (num_points_in_pop + num_other_points_in_hull)
				frac_sum += frac_other_pop_points_in_hull

				if plot:
					hull = ConvexHull(this_pop_coords)
					plt.scatter(this_pop_coords[:,0], this_pop_coords[:,1], color="red")
					plt.scatter(other_pop_coords[:,0], other_pop_coords[:,1], color="black", alpha=0.8)
					for simplex in hull.simplices:
						plt.plot(this_pop_coords[simplex, 0], this_pop_coords[simplex, 1], 'k-')
					plt.show()
					plt.close()

			else:
				print("Too few for hull: " + str(pop) )
				if num_points_in_pop < 3:
					print("--  shape: " + str(this_pop_coords.shape) + ": skipping")
					continue
				elif num_points_in_pop == 3:
					n_dim = int(num_points_in_pop) - 1
				else:
					n_dim = int(num_points_in_pop) - 2

				print("--  shape: " + str(this_pop_coords.shape) + ": using " + str(n_dim) + " dimensions instead")

				num_other_points_in_hull = sum(in_hull(other_pop_coords[:,0:n_dim], this_pop_coords[:,0:n_dim]))
				frac_other_pop_points_in_hull = num_other_points_in_hull / (num_points_in_pop + num_other_points_in_hull)
				frac_sum += frac_other_pop_points_in_hull


		hull_error = frac_sum / num_pops

	except Exception as e:

		print("Exception in calculating hull error: {0}".format(e))
		traceback.print_exc(file=sys.stdout)
		hull_error = -1.0

	return hull_error


def get_baseline_gc(genotypes):
	'''
	Get the genotype concordance of guessing the most frequent genotype per SNP for every sample.

	:param genotypes: n_samples x n_genos array of genotypes coded as 0,1,2
	:return: genotype concordance value
	'''

	n_samples = float(genotypes.shape[0])
	modes = stats.mode(genotypes)
	most_common_genos = modes.mode
	counts = modes.count
	# num with most common geno / total samples gives correctness when guessing most common for every sample
	genotype_conc_per_marker = counts/n_samples
	genotype_conc = np.average(genotype_conc_per_marker)
	return genotype_conc



def get_pops_with_k(k, coords_by_pop):
	'''
	Get a list of unique populations that have at least k samples
	:param coords_by_pop: dict mapping pop names to a list of coords for each sample of that pop
	:return: list
	'''
	res = []
	for pop in coords_by_pop.keys():
		if len(coords_by_pop[pop]) >= k:
			res.append(pop)
		else:
			try:
				print("-- {0}".format(pop.decode("utf-8")))
			except:
				print("-- {0}".format(pop))

	return res


def f1_score_kNN(x, labels, labels_to_use, k = 5):
	classifier = KNeighborsClassifier(n_neighbors=k)
	classifier.fit(x, labels)
	predicted_labels = classifier.predict(x)
	# this returns a vector of f1 scores per population
	f1_score_per_pop = f1_score(y_true=labels, y_pred=predicted_labels, labels = labels_to_use, average = None)
	f1_score_avg = f1_score(y_true=labels, y_pred=predicted_labels, average = "micro")


	return f1_score_avg, f1_score_per_pop


def my_tf_round(x, d = 2, base = 0.5):
	'''
	Round input to nearest multiple of base, considering d decimals.

	:param x: tensor
	:param d: number of decimals to consider
	:param base: rounding to nearest base
	:return: x rounded
	'''
	multiplier = tf.constant(10 ** d, dtype=x.dtype)
	x = base * tf.math.round(tf.math.divide(x,base))
	return tf.math.round(x * multiplier) / multiplier


def to_genotypes_sigmoid_round(data):
	'''
	Interpret data as genotypes by applying sigmoid
	function and rounding result to closest of 0.0, 0.5, 1.0

	:param data: n_samples x n_markers
	:return: data transformed
	'''
	data = tf.keras.activations.sigmoid(data)
	data = tf.map_fn(my_tf_round, data)
	return data


def to_genotypes_invscale_round(data, scaler_vals):
	'''
	Interpret data as genotypes by applying inverse scaling
	based on scaler_vals, and rounding result to closest integer.

	:param data: n_samples x n_markers
	:param scaler_vals tuple of means and 1/stds that were used to scale the data.
	:return: data transformed
	'''

	means = scaler_vals[0]
	stds = scaler_vals[1]
	genos = data.T

	for m in range(len(genos)):
		genos[m] = np.add(np.multiply(genos[m],stds[m]), means[m])

	output = tf.map_fn(lambda x : my_tf_round(x, base = 1.0), genos.T)

	return output


class GenotypeConcordance(keras.metrics.Metric):
	'''
	Genotype concordance metric.

	Assumes pred and true are genotype values scaled the same way.

	'''
	def __init__(self, name='genotype_concordance', **kwargs):
		super(GenotypeConcordance, self).__init__(name=name, **kwargs)
		self.accruary_metric = tf.keras.metrics.Accuracy()

	def update_state(self, y_true, y_pred, sample_weight=None):
		_ = self.accruary_metric.update_state(y_true=y_true, y_pred = y_pred)
		return y_pred

	def result(self):
		return self.accruary_metric.result()

	def reset_states(self):
		# The state of the metric will be reset at the start of each epoch.
		self.accruary_metric.reset_states()


def write_h5(filename, dataname, data, replace_file = False):
	'''
	Write data to a h5 file.
	Replaces dataset dataname if already exists.
	If replace_file specified: skips writing data if replacing the dataset fails.

	:param filename: directory and filename (with .h5 extension) of file to write to
	:param dataname: name of the datataset
	:param data: the data
	:param replace_file: if replacing existing dataset does not work: overwrite entire file with new data,
						 all old data is lost

	'''
	try:
		with h5py.File(filename, 'a') as hf:
			try:
				hf.create_dataset(dataname,  data = data)
			except (RuntimeError, ValueError):
				print("Replacing dataset {0} in {1}".format(dataname, filename))
				del hf[dataname]
				hf.create_dataset(dataname,  data = data)
	except OSError:
		print("Could not write to file {0}.".format(filename))
		if replace_file:
			print("Replacing {0}.".format(filename))
			with h5py.File(filename, 'w') as hf:
				try:
					hf.create_dataset(dataname, data = data)
				except RuntimeError:
					print("Could not replace {0}. Data not written.".format(filename))


def read_h5(filename, dataname):
	'''
	Read data from a h5 file.

	:param filename: directory and filename (with .h5 extension) of file to read from
	:param dataname: name of the datataset in the h5
	:return the data
	'''
	with h5py.File(filename, 'r') as hf:
		data = hf[dataname][:]
	return data


def get_pop_superpop_list(file):
	'''
	Get a list mapping populations to superpopulations from file.

	:param file: directory, filename and extension of a file mapping populations to superpopulations.
	:return: a (n_pops) x 2 list

	Assumes file contains one population and superpopulation per line, separated by ","  e.g.

	Kyrgyz,Central/South Asia
	Khomani,Sub-Saharan Africa

	'''

	pop_superpop_list = np.genfromtxt(file, usecols=(0,1), dtype=str, delimiter=",")
	return pop_superpop_list


def get_ind_pop_list_from_map(famfile, mapfile):
	'''
	Get a list of individuals and their populations from
	a .fam file that contains individual IDs, and
	a .map file that maps individual IDs to populations.

	The order of the individuals in the mapfile does not have to be the same as in the famfile.
	The order of indiivudals in the output will be the same as in the famfile.

	:param famfile:
	:param mapfile:

	:return: (n_samples x 2) array of ind_id, pop_id
	'''
	try:
		ind_list = np.genfromtxt(famfile, usecols=(1), dtype=str)
		print("Reading ind list from {0}".format(famfile))
		ind_pop_map_list = np.genfromtxt(mapfile, usecols=(0,2), dtype=str)
		print("Reading ind pop map from {0}".format(mapfile))

		ind_pop_map = dict()
		for ind, pop in ind_pop_map_list:
			ind_pop_map[ind] = pop

		ind_pop_list = []
		for ind in ind_list:
			ind_pop_list.append([ind, ind_pop_map[ind]])

		return np.array(ind_pop_list)

	except Exception as e:
		exc_str = traceback.format_exc()
		print("Error in gettinf ind pop list from map : {0}".format(exc_str))


def get_ind_pop_list(filestart):
	'''
	Get a list of individuals and their populations from a .fam file.
	or if that does not exist, tried to find a a .ind file

	:param filestart: directory and file prefix of file containing sample info
	:return: an (n_samples)x(2) list where ind_pop_list[n] = [individial ID, population ID] of the n:th individual
	'''
	try:
		ind_pop_list = np.genfromtxt(filestart + ".fam", usecols=(1,0), dtype=str)
		print("Reading ind pop list from " + filestart + ".fam")
	except:
		ind_pop_list = np.genfromtxt(filestart+".ind", usecols=(0,2), dtype=str)
		print("Reading ind pop list from " + filestart + ".ind")

		# probably not general solution
		if ":" in ind_pop_list[0][0]:
			nlist = []
			for v in ind_pop_list:
				print(v)
				v = v[0]
				nlist.append(v.split(":")[::-1])
			ind_pop_list = np.array(nlist)
	return ind_pop_list

def get_unique_pop_list(filestart):
	'''
	Get a list of unique populations from a .fam file.

	:param filestart: directory and file prefix of file containing sample info
	:return: an (n_pops) list where n_pops is the number of unique populations (=families) in the file filestart.fam
	'''

	pop_list = np.unique(np.genfromtxt(filestart + ".fam", usecols=(0), dtype=str))
	return pop_list


def get_coords_by_pop(filestart_fam, coords, pop_subset = None, ind_pop_list = []):
	'''
	Get the projected 2D coordinates specified by coords sorted by population.

	:param filestart_fam: directory + filestart of fam file
	:param coords: a (n_samples) x 2 matrix of projected coordinates
	:param pop_subset: list of populations to plot samples from, if None then all are returned
	:param ind_pop_list: if specified, gives the ind and population IDs for the samples of coords. If None: assumed that filestart_fam has the correct info.
	:return: a dict that maps a population name to a list of list of 2D-coordinates (one pair of coords for every sample in the population)


	Assumes that filestart_fam.fam contains samples in the same order as the coordinates in coords.

	'''

	try:
		new_list = []
		for i in range(len(ind_pop_list)):
			new_list.append([ind_pop_list[i][0].decode('UTF-8'), ind_pop_list[i][1].decode('UTF-8')])
		ind_pop_list = np.array(new_list)
	except:
		pass

	if not len(ind_pop_list) == 0:
		unique_pops = np.unique(ind_pop_list[:,1])

	else:
		ind_pop_list = get_ind_pop_list(filestart_fam)
		unique_pops = get_unique_pop_list(filestart_fam)

	pop_list = ind_pop_list[:,1]

	coords_by_pop = {}
	for p in unique_pops:
		coords_by_pop[p] = []

	for s in range(len(coords)):
		this_pop = pop_list[s]
		this_coords = coords[s]
		if pop_subset is None:
			coords_by_pop[this_pop].append(this_coords)
		else:
			if this_pop in pop_subset:
				coords_by_pop[this_pop].append(this_coords)

	return coords_by_pop


def get_saved_epochs(train_directory):
	'''
	Get an ordered list of the saved epochs in the given directory.

	:param train_directory: directory where training data is stored
	:return: int list of sorted epochs
	'''
	epochs = []
	for i in os.listdir(train_directory+"/weights"):
		start = i.split("/")[-1].split(".")[0]
		try:
			num = int(start)
			if not num in epochs:
				epochs.append(num)
		except:
			continue

	epochs.sort()

	return epochs


def get_projected_epochs(encoded_data_file):
	'''
	Get an ordered list of the saved projected epochs in encoded_data_file.

	:param encoded_data_file: h5 file of encoded data
	:return: int list of sorted epochs
	'''

	epochs = []

	if os.path.isfile(encoded_data_file):
		encoded_data = h5py.File(encoded_data_file, 'r')

		for i in encoded_data.keys():
			start = i.split("_")[0]
			try:
				num = int(start)
				if not num in epochs:
					epochs.append(num)
			except:
				continue

		epochs.sort()

	else:
		print("Encoded data file not found: {0} ".format(encoded_data_file))

	return epochs

def write_metric_per_epoch_to_csv(filename, values, epochs):
	'''
	Write value of a metric per epoch to csv file, extending exisitng data if it exists.

	Return the total data in the file, the given values and epochs appended to
	any pre-existing data in the file.

	Assumes format of file is epochs on first row, corresponding values on second row.

	:param filename: full name and path of csv file
	:param values: array of metric values
	:param epochs: array of corresponding epochs
	:return: array of epochs appended to any pre-existing epochs in filename
			 and
			 array of metric values appended to any pre-existing values in filename
	'''
	epochs_saved = np.array([])
	values_saved = np.array([])

	try:
		with open(filename, mode='r') as res_file:
			res_reader = csv.reader(res_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
			epochs_saved = next(res_reader)
			values_saved = next(res_reader)
	except:
		pass

	epochs_combined = np.concatenate((epochs_saved, epochs), axis=0)
	values_combined = np.concatenate((values_saved, values), axis=0)

	with open(filename, mode='w') as res_file:
		res_writer = csv.writer(res_file, delimiter=',')
		res_writer.writerow(epochs_combined)
		res_writer.writerow(np.array(values_combined))

	return  epochs_combined, values_combined


def plot_genotype_hist(genotypes, filename):
	'''
	Plots a histogram of all genotype values in the flattened genotype matrix.

	:param genotypes: array of genotypes
	:param filename: filename (including path) to save plot to
	'''
	unique, counts = np.unique(genotypes, return_counts=True)
	d = zip(unique, counts)
	plt.hist(np.ndarray.flatten(genotypes), bins=50)
	if len(unique) < 5:
		plt.title(", ".join(["{:.2f} : {}".format(u, c) for (u,c) in d]), fontdict = {'fontsize' : 9})

	plt.savefig("{0}.pdf".format(filename))
	plt.close()


def get_superpop_pop_dict(pop_superpop_file):
	'''
	Get a dict mapping superpopulation IDs to a list of their population ID

	Assumes file contains one population and superpopulation per line, separated by ","  e.g.

	Kyrgyz,Central/South Asia

	Khomani,Sub-Saharan Africa

	:param pop_superpop_file: name of file mapping populations to superpopulations
	:return: a dictionary mapping each superpopulation ID in the given file to a list of its subpopulations
	'''
	pop_superpop_list = get_pop_superpop_list(pop_superpop_file)
	superpops = np.unique(pop_superpop_list[:,1])

	superpop_pop_dict = {}
	for i in superpops:
		superpop_pop_dict[i] = []

	for pp in pop_superpop_list:
		pop = pp[0]
		superpop = pp[1]
		superpop_pop_dict[superpop].append(pop)

	return superpop_pop_dict


def genfromplink(fileprefix):
	'''
	Generate genotypes from plink data.
	Replace missing genotypes by the value 9.0.

	:param fileprefix: path and filename prefix of the plink data (bed, bim, fam)
	:return:
	'''
	(bim, fam, bed) = read_plink(fileprefix)
	genotypes = bed.compute()
	genotypes[np.isnan(genotypes)] = 9.0
	return (genotypes, bed.shape[0])



def get_test_samples_stratified(genotypes, ind_pop_list, test_split):
	'''
	Generate a set of samples stratified by population from eigenstratgeno data.

	Samples from populations with only one sample are considered as belonging to the same temporary population,
	and stratified according to that. If there is only one such sample, another one is randomly assigned the
	temporary population.

	:param genotypes: (n_samples x n_markers) array of genotypes
	:param ind_pop_list: (n_smaples x 2) list of individual id and population of all samples
	:param test_split: fraction of samples to include in test set
	:return: genotypes_train (n_train_samples x n_markers): genotypes of train samples
			 genotypes_test (n_test_samples x n_markers): genotypes of test samples
			 sample_idx_train (n_train_samples): indices of train samples in original data
			 sample_idx_test (n_test_samples): indices of test samples in original data


	If test_split is 0.0, genotypes_test and sample_idx_test are [].
	'''
	pop_list = np.array(ind_pop_list[:,1])
	panel_pops = np.unique(pop_list)
	n_samples = len(pop_list)


	if test_split == 0.0:
		return genotypes, [], range(len(ind_pop_list)), []

	################### stratify cant handle classes with only one sample ########################

	else:
		counter_dict = {}
		for pop in panel_pops:
			counter_dict[pop] = 0

		for ind,pop in ind_pop_list:
			counter_dict[pop] += 1

		for pop in counter_dict.keys():
			if counter_dict[pop] == 1:
				np.place(pop_list, pop_list == pop, ["Other"])

		# If we only have one sample with "Other": randomly select another one so its at least 2 per pop.
		if len(np.where(pop_list == "Other")[0]) == 1:
			r = np.random.choice(range(n_samples))
			while r in np.where(pop_list == "Other")[0]:
				r = np.random.choice(range(n_samples))
			pop_list[r] = "Other"


	################################################################################################
		panel_pops = np.unique(pop_list)
		n_pops = len(panel_pops)

		if np.ceil(test_split * n_samples) < n_pops:
			test_split = float(n_pops) / n_samples
			print(str(test_split * n_samples) + "{0} is too few samples for " + str(n_pops) +" classes. Setting split fraction to " + str(test_split))


		sample_idx = range(len(genotypes))
		genotypes_train, genotypes_test, sample_idx_train, sample_idx_test, pops_train, pops_test = train_test_split(genotypes, sample_idx, pop_list, test_size=test_split, stratify=pop_list, random_state = 0)
		sample_idx_train = np.array(sample_idx_train)
		sample_idx_test = np.array(sample_idx_test)

		pops_train_recon = pop_list[sample_idx_train]
		pops_test_recon = pop_list[sample_idx_test]

		for i in range(len(pops_train_recon)):
			assert pops_train_recon[i] == pops_train[i]
		for i in range(len(pops_test_recon)):
			assert pops_test_recon[i] == pops_test[i]

		return genotypes_train, genotypes_test, sample_idx_train, sample_idx_test




def get_test_samples_stratified2(ind_pop_list, test_split):
	'''
	Generate a set of samples stratified by population from eigenstratgeno data.
	FILIP EDIT: REMOVED THE GENOTYPE INPUT/OUTPUT- CANT FIND THE UISE OF IT. ONLY INDICES ARE OF INTEREST
	Samples from populations with only one sample are considered as belonging to the same temporary population,
	and stratified according to that. If there is only one such sample, another one is randomly assigned the
	temporary population.

	:param genotypes: (n_samples x n_markers) array of genotypes
	:param ind_pop_list: (n_smaples x 2) list of individual id and population of all samples
	:param test_split: fraction of samples to include in test set
	:return: genotypes_train (n_train_samples x n_markers): genotypes of train samples
			 genotypes_test (n_test_samples x n_markers): genotypes of test samples
			 sample_idx_train (n_train_samples): indices of train samples in original data
			 sample_idx_test (n_test_samples): indices of test samples in original data


	If test_split is 0.0, genotypes_test and sample_idx_test are [].
	'''
	pop_list = np.array(ind_pop_list[:,1])
	panel_pops = np.unique(pop_list)
	n_samples = len(pop_list)


	if test_split == 0.0:
		return  range(len(ind_pop_list)), []

	################### stratify cant handle classes with only one sample ########################

	else:
		counter_dict = {}
		for pop in panel_pops:
			counter_dict[pop] = 0

		for ind,pop in ind_pop_list:
			counter_dict[pop] += 1

		for pop in counter_dict.keys():
			if counter_dict[pop] == 1:
				np.place(pop_list, pop_list == pop, ["Other"])

		# If we only have one sample with "Other": randomly select another one so its at least 2 per pop.
		if len(np.where(pop_list == "Other")[0]) == 1:
			r = random.choice(range(n_samples))
			while r in np.where(pop_list == "Other")[0]:
				r = random.choice(range(n_samples))
			pop_list[r] = "Other"


	################################################################################################
		panel_pops = np.unique(pop_list)
		n_pops = len(panel_pops)

		if np.ceil(test_split * n_samples) < n_pops:
			test_split = float(n_pops) / n_samples
			print(str(test_split * n_samples) + "{0} is too few samples for " + str(n_pops) +" classes. Setting split fraction to " + str(test_split))


		sample_idx = range(len(pop_list))
		sample_idx_train, sample_idx_test, pops_train, pops_test = train_test_split(sample_idx, pop_list, test_size=test_split, stratify=pop_list)
		sample_idx_train = np.array(sample_idx_train)
		sample_idx_test = np.array(sample_idx_test)

		pops_train_recon = pop_list[sample_idx_train]
		pops_test_recon = pop_list[sample_idx_test]

		for i in range(len(pops_train_recon)):
			assert pops_train_recon[i] == pops_train[i]
		for i in range(len(pops_test_recon)):
			assert pops_test_recon[i] == pops_test[i]

		return sample_idx_train, sample_idx_test



def get_most_frequent(data):
	'''
	Get the most frequently occurring value in data along axis 0.
	If two or more values occur the same number of times, take the smallest of them.

	:param data: the data
	:return:
	'''

	modes = stats.mode(data)
	most_common_feature = modes.mode
	return most_common_feature[0]



class alt_data_generator(data_generator_ae):

	def __init__(self, filebase,
		 		 batch_size, normalization_mode = "genotypewise01",
		  		normalization_options ={"flip": False, "missing_val":-1.0, "num_markers":None}, impute_missing = True, sparsifies = False ):

		self.filebase = filebase 
		self.batch_size = batch_size
		self._define_samples()
		self.normalization_mode = normalization_mode
		self.normalization_options = normalization_options
		self.impute_missing = impute_missing
		self.sparsifies = sparsifies
		self.missing_val = normalization_options["missing_val"]

		self.marker_count() # This sets the number of markers
		#if normalization_options["num_markers"] is not None:
		self.n_markers = normalization_options["num_markers"]
		print(self.n_markers)

		if  sparsifies == False:
			self.sparsify_input = False
		else:
			self.sparsify_input = True


	def generator(self, pref_chunk_size, training, shuffle = True):

		"""
			This function reads a chunk of data, which I define as an integer multiple of the batch size, into memory.
			It defines a generator, that we can then feed into Tensorflow and create a dataset from it by calling
			ds = tf.data.Dataset.from_generator("this_function").

			This enables us to read large files, without dumping all the data to memory (I hope).
			Also, since the data is huge, reading the whole dataset takes a lot of time.
			This way, we can start training earlier, not having to wait for all samples to be loaded in before training.
			Using the TensorFlow data API, we can then effectively prefetch, and apply the normalization and sparsification
			as maps, which can be computed in parallel.

			There may of course be stuff I have done incorrectly, or understood wrongly.

			As of right now, I am using .parquet files. This may not be optimal, but the format is very efficient at loading selected columns (samples)
			from the data set, and does not have to go through all the samples.

			param: training, if True, draw training samples, if False, draw the valdiation points

		"""

		if training:
			training = True
			n_samples = self.n_train_samples
			# This yields randomized chunks and batches for each epoch. Essentially shuffling the samples
			if shuffle:
				indices = self.sample_idx_train[np.random.choice(n_samples, n_samples, replace=False)]
			else:
				indices = self.sample_idx_train[np.arange(0,n_samples)]
				

		elif not training:
			training = False
			n_samples = self.n_valid_samples
			if shuffle:
				indices = self.sample_idx_valid[np.random.choice(n_samples, n_samples, replace=False)]
			else:
				indices = self.sample_idx_valid[np.arange(0,n_samples)]
			

		chunk_size = pref_chunk_size // self.batch_size * self.batch_size # Make the preferred chunk size an integer multiple of the batch size
		num_chunks = np.ceil(n_samples / chunk_size)
		chunks_read = 0

		while chunks_read < num_chunks:
			chunk_indices = indices[chunk_size * chunks_read: chunk_size * (chunks_read + 1)]
			chunk_inds = ["".join(item) for item in chunk_indices.astype(str)] # The indices needs to be given as a list of strings for the parquet-file read.

			df = pd.read_parquet(self.filebase + ".parquet", columns=chunk_inds) 
			# Read only the selected indices into memory,
			# this read is made to be as large as possible, consisting of several batches, but (perhaps not the entire dataset)

			df_numpy = pd.DataFrame.to_numpy(df)
			df_numpy = df_numpy[0:self.n_markers, : ]

			batches_per_chunk = np.ceil(len(chunk_indices) / self.batch_size)
			# Set nan values to 9. 
			df_numpy[np.where(np.isnan(df_numpy))] = 9
			batches_read = 0
			while batches_read < batches_per_chunk:

				genotypes_train = df_numpy[:, self.batch_size * batches_read: self.batch_size * (batches_read + 1)]

				inds = chunk_indices[self.batch_size * batches_read: self.batch_size * (batches_read + 1)]

				# If the shape does not match the usual batch size, i.e., we are on the last batch, also yield the keyword "last_batch" as a True bool. Need to explicitly give this since
				# Tensorflow does not know the actual batch size at run time, and I need to create the mask not having access to .shape commands at run time.
				
				if genotypes_train.shape[1] == self.batch_size:
					yield genotypes_train, self.ind_pop_list_train_orig[inds], [False, training]
				else:
					yield genotypes_train, self.ind_pop_list_train_orig[inds], [True,  training]

				batches_read += 1
			chunks_read += 1
			
	def normalize_mapping(self, x, inds, args):
		"""
			The purpose of this function is to use as mapping onto a dataset.
			It returns the data normalized, and transposes the matrix into the desired shape of (n_samples, n_markers).
			The specific normalization is to be specified in the run_gcae - call.

			The args input is a list of [last_batch, training] both bools, indicating whether or the current batch is the last one (with different size)
			and if it is the training set or vcalidation set.
		"""
		last_batch = args[0]
		training = args[1]
		# assuming the missing values are 9
		missing_indices = tf.where(tf.transpose(x) == 9)
		
		if last_batch == False:

			num_samples = self.batch_size
		else:
		
			if training : num_samples = self.n_train_samples_last_batch
			else : num_samples = self.n_valid_samples_last_batch
	
		# a is a sparse tensor containing the missing values.
		a = tf.sparse.SparseTensor(indices=missing_indices,
								values=tf.ones(shape=tf.shape(missing_indices)[0], dtype=tf.float32),
								dense_shape=(num_samples, self.n_markers))

		indices = missing_indices[:,1]
		x  = tf.transpose(x)
		if self.impute_missing: 
			b = tf.sparse.SparseTensor(indices=missing_indices,
										values=(tf.gather(self.most_common_genos,indices = indices) -9),
										dense_shape=(num_samples, self.n_markers))
			x = tf.sparse.add(x,b)
			
		if self.normalization_mode == "genotypewise01":
			if self.normalization_options["flip"]:
				# This operation results in the missing values having a value of -7/2, wrong! Amend this by adding 3.5-missing_val
				x = -(x - 2) / 2
				if not self.impute_missing:
					x = tf.sparse.add(x,a * (3.5 - self.missing_val))
			else:
				# This operation results in the missing values having a value of 4.5, wrong! Amend this by subtracting 4.5-missing_val
				x = x / 2
				if not self.impute_missing:
					x = tf.sparse.add(x, a  * ( self.missing_val - 4.5))

		elif self.normalization_mode == "standard" or self.normalization_mode == "smartPCAstyle":
			if self.normalization_options["flip"]:
				x = -(x - 2)
			x2 = (x - self.scaler.mean_) / np.sqrt(self.scaler.var_)
			if not self.impute_missing:
				x = tf.sparse.add(x2, a.__mul__(self.missing_val - x2))
				
		return x, inds, last_batch

	def sparse_mapping(self, x, inds, last_batch):
		"""
		I am having a hard time getting the shapes correct for the case with no sparsification.
		It says that it expects [batch_size, n_markers, 1], but receives [batch_size, n_markers]. I have not yet come up with a solution for this.
		In the meantime, if we want to run with no sparsification, just set sparsifies = [0] in the data_ops files.

		"""
		if self.sparsify_input and self.training:
			# Not 100% straight forward how to get the same 'rolling sparsification' that the sparsify fraction uses the next value for the next batch.
			# Here I am just choosing one of the fractions at random. Should have essentially the same effect if I am not missing anything.
			# Makes it hard to compare the runs for the original code vs my alteration. 
			#sparsify_fraction = self.sparsifies[np.random.choice(len(self.sparsifies))]
			sparsify_fraction = tf.random.shuffle(self.sparsifies)[0]
			#sparsify_fraction = self.sparsifies[tf.random.uniform(shape=(1,), minval = 0, maxval = len(self.sparsifies), dtype=tf.int32 )]
		else:
			sparsify_fraction = 0.0
		try:
			if self.project:
				sparsify_fraction = 0.0
		except:
			pass

		missing_value = self.missing_val

		if last_batch:
			if self.training:
				mask = tf.experimental.numpy.full(shape=(self.n_train_samples_last_batch, self.n_markers), fill_value=1.0,
								dtype=tf.float32)
				b = tf.random.uniform(shape = (self.n_train_samples_last_batch, self.n_markers),minval = 0, maxval = 1)

				indices = tf.where(b<sparsify_fraction)
								
				b = tf.sparse.SparseTensor(indices=indices,
														values=(tf.repeat(-1.0, tf.shape(indices)[0])),
														dense_shape= (self.n_train_samples_last_batch, self.n_markers))
				mask = tf.sparse.add(mask,b)
			
			else:
			
				mask = tf.experimental.numpy.full(shape=(self.n_valid_samples_last_batch, self.n_markers), fill_value=1.0,
								dtype=tf.float32)
				b = tf.random.uniform(shape = (self.n_valid_samples_last_batch, self.n_markers),minval = 0, maxval = 1)

				indices = tf.where(b<sparsify_fraction)
								
				b = tf.sparse.SparseTensor(indices=indices,
														values=(tf.repeat(-1.0, tf.shape(indices)[0])),
														dense_shape= (self.n_valid_samples_last_batch, self.n_markers))
				mask = tf.sparse.add(mask,b)

		else:
		
			mask = tf.experimental.numpy.full(shape=(self.batch_size, self.n_markers), fill_value=1.0,
							dtype=tf.float32)
			b = tf.random.uniform(shape = (self.batch_size, self.n_markers),minval = 0, maxval = 1)
			indices = tf.where(b<sparsify_fraction)			
			b = tf.sparse.SparseTensor(indices=indices,
													values=(tf.repeat(-1.0, tf.shape(indices)[0])),
													dense_shape= (self.batch_size, self.n_markers))
			mask = tf.sparse.add(mask,b)
		
		sparsified_data = tf.math.add(tf.math.multiply(x, mask), -1 * missing_value * (mask - 1))
		#tf.print(tf.executing_eagerly())
		input_data_train = tf.stack([sparsified_data, mask], axis=-1)

		if self.missing_mask_input:
			return input_data_train, x, inds

		else:
			# Just extract everything but the last dimension, which contains the mask.
			return input_data_train[:, :, 0], x, inds

	def marker_count(self):
		"""
		Just a funtion to count the number of markers.
		Reads only one column, does not have to read entire dataset
		"""
		df = pd.read_parquet(self.filebase+".parquet", columns=["0"])
		self.n_markers = len(df.to_numpy())

	def prep_normalizer(self, ds):
		# TODO- Fix so that the normalizer is calibrated with more samples than it is trained with. In this case, if very few samples are used, there may be some issues if all have missing data at the same marker position
		temp = np.zeros((self.n_markers,3))

		print("Calibrating data scaler for normalization, remember to fix this. In prep_normalizer. Note in the function def.")
		if self.normalization_mode == "standard" or self.normalization_mode == "smartPCAstyle" or self.impute_missing == True:
			self.scaler = StandardScaler()

			for i, j, _ in ds:
				a = np.zeros(shape=i.shape)
				a[i == 9] = np.nan

				if self.normalization_options["flip"]:
					i = -(i - 2)  # This sends 1 to 1, 0 to 2, and 2 to 0
				self.scaler.partial_fit(tf.transpose(a + i))
				

				for k in range(3):

					temp[:,k] += tf.reduce_sum(tf.cast(i == k, tf.int32),axis = 1) 
			

			self.most_common_genos = tf.cast(tf.transpose(np.argmax(temp, axis = 1)),tf.float32)

			if self.normalization_mode == "smartPCAstyle":
				p = (1 + self.scaler.mean_ * self.scaler.n_samples_seen_) / (2 + 2 * self.scaler.n_samples_seen_)

				self.scaler.var_ = p * (1 - p)


	def create_dataset(self,pref_chunk_size, mode, shuffle = True):
		if mode ==  "training":
			self.training = True
		elif mode == "validation":
			self.training = False
		else:
			print("pass either training or validation")
			return -1

		ds = tf.data.Dataset.from_generator(self.generator,
											output_signature=(tf.TensorSpec(shape=(self.n_markers, None), dtype=tf.float32),
																tf.TensorSpec(shape=(None, 2), dtype=tf.string),
																tf.TensorSpec(shape=(2,), dtype=tf.bool)),
											args=[pref_chunk_size, self.training, shuffle])
		if mode == "training":
			self.prep_normalizer(ds)  # This generates the needed scalers for normalization´using standard and smartPCAstyle
		options = tf.data.Options()
		
		options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		ds  = ds.with_options(options)
	

		ds = ds.prefetch(tf.data.AUTOTUNE).map(self.normalize_mapping, num_parallel_calls=tf.data.AUTOTUNE)
		ds = ds.map(self.sparse_mapping, num_parallel_calls=tf.data.AUTOTUNE)
	
		
		return ds


def parquet_converter(filebase, max_mem_size):
	"""

	This function is made so that we can use .parquet files in the input pipeline, without having the user make sure that the correct fileformat exists.

	I have also attempted to make sure that the function can convert datasets that are too large to fit in memory in its entirety.
	The parameter max_mem_size regulates how many large chunks we can read and write at a time. It is to be given in Bytes,
	so say that we want to set a cap at 10GB set max_mem_size to 10^10.
	
	I think it is good to take as large a max_mem_size as possible, to achieve maximal compression in the resulting combined parquet file

	Right now I am only supporting the input as a PLINK format. The eigenstratgeno is very slow to load, and can really only load entire dataset I think.


	"""

	# First, check if there already exists a correct parquet file, if it does just return without doing anything

	if os.path.isfile(filebase + ".parquet"):
		print("Not creating parquet file: Already exists.")
		return
	
	print("Creating .parquet file from PLINK for faster data extraction. This may take a while")

	snpreader = Bed(filebase + ".bed", count_A1=True)
	# Step 1, partition the dataset indices according to the maximum readable amount
	# The shape of the .bed files are given as (n_samples x n_markers), read as dtype = float32 = 8 Bytes
	
	(n_samples, n_markers) = snpreader.shape
	
	n_samples_in_chunk = np.floor(max_mem_size / 8.0 / n_samples)

	n_chunks = np.ceil(n_markers / n_samples_in_chunk)

	# Step 2, While looping over the chunks of indices, read the data into numpy arrays

	if not os.path.isdir("Data_temp"):
		os.mkdir("Data_temp")

	for i in range(int(n_chunks)):
		a = snpreader[:, int(n_samples_in_chunk * i):int(n_samples_in_chunk * (i + 1))].read().val

		# Step 3, Save each ndarray into .parquet file
		df = pd.DataFrame(a.T)
		table = pa.Table.from_pandas(df)
		it_number = 8 - len(str(i))
		if n_chunks > 1:
			pq.write_table(table, "Data_temp/temp" + it_number * "0" + str(i) + ".parquet")
		else: 
			pq.write_table(table, filebase+".parquet")

	if n_chunks > 1:
		# Step 4, Combine smaller files into 1 large parq
		def combine_parquet_files(input_folder, target_path):
			try:

				file_name = os.listdir(input_folder)[0]
				temp_data = pq.read_table(os.path.join(input_folder, file_name))

				with pq.ParquetWriter(target_path,
									temp_data.schema,
									version='2.6',
									compression='gzip',
									use_dictionary=True,
									data_page_size=2097152,  # 2MB
									write_statistics=True) as writer:
					for file_name in os.listdir(input_folder):
						writer.write_table(pq.read_table(os.path.join(input_folder, file_name)))
			except Exception as e:
				print(e)

		combine_parquet_files('Data_temp', 'Data/combined.parquet')

	# step 5, delete any temp files
	shutil.rmtree("Data_temp")





