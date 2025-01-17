import os
import numba
import numpy as np
import h5py
import utils.normalization as normalization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
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
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pyarrow.parquet as pq
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pysnptools.snpreader import Bed
import pyarrow as pa
import shutil
import time


@numba.jit(nopython=True, parallel=True)
def helper_get_train_batch(sparsify, n_samples_batch, genotypes_train, genotypes_train_orig, mask_train,
                           indices_this_batch, missing_val):
    input_data_train = np.full((n_samples_batch, genotypes_train.shape[1], 2), 1.0, dtype=np.dtype('f4'))
    # genotypes_train = np.copy(self.genotypes_train_orig[self.sample_idx_train[indices_this_batch]])
    #

    input_data_train[:, :, 0] = genotypes_train
    input_data_train[:, :, 1] = mask_train

    targets = genotypes_train_orig
    return input_data_train, targets


class data_generator_ae:
    '''
    Class to generate data for training and evaluation.

    If get_data is False then only ind pop list will be generated

    '''

    def __init__(self,
                 filebase,
                 normalization_mode="smartPCAstyle",
                 normalization_options={"flip": False, "missing_val": 0.0, "num_markers": 100},
                 get_genotype_data=True,
                 impute_missing=True):
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
            self._normalize(normalization_options["num_markers"])

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

    def _normalize(self, num_markers):
        '''
        Normalize the genotype data.

        '''

        ind_pop_list = get_ind_pop_list(self.filebase)
        n_samples = len(ind_pop_list)

        try:
            genotypes = np.genfromtxt(self.filebase + ".eigenstratgeno", delimiter=np.repeat(1, n_samples))
            genotypes = genotypes[0:num_markers, :]
            self.n_markers = len(genotypes)
        except:
            (genotypes, self.n_markers) = genfromplink(self.filebase)

        if self.impute_missing:
            self._impute_missing(genotypes)

        genotypes = np.array(genotypes,
                             order='F')  # Cheeky, this will then be transposed, so we have individual-major order

        genotypes_train = genotypes[:, self.sample_idx_all]

        normalization_method = getattr(normalization, "normalize_genos_" + self.normalization_mode)
        no2 = dict(self.normalization_options)
        del no2["num_markers"]

        genotypes_train_normed, _, scaler = normalization_method(genotypes_train, np.array([]),
                                                                 get_scaler=True,
                                                                 **no2)
        self.scaler = scaler

        self.genotypes_train_orig = np.array(genotypes_train_normed, dtype=np.dtype('f4'), order='C')

    def get_nonnormalized_data(self):
        '''
        Get the non-nornalized training data.
        Missing data represented by missing_val.

        :return: train data (n_samples x n_markers)

        '''
        ind_pop_list = get_ind_pop_list(self.filebase)
        n_samples = len(ind_pop_list)

        try:
            genotypes = np.genfromtxt(self.filebase + ".eigenstratgeno", delimiter=np.repeat(1, n_samples))
            self.n_markers = len(genotypes)
        except:
            (genotypes, self.n_markers) = genfromplink(self.filebase)

        if self.impute_missing:
            self._impute_missing(genotypes)
        else:
            genotypes[genotypes == 9.0] = self.missing_val

        genotypes_train = np.array(genotypes[:, self.sample_idx_train].T, order='C')

        return genotypes_train

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

        self.sample_idx_train, self.sample_idx_valid = get_test_samples_stratified2(self.ind_pop_list_train_orig,
                                                                                    validation_split)

        self.sample_idx_train = np.array(self.sample_idx_train)
        self.sample_idx_valid = np.array(self.sample_idx_valid)

        self.train_set_indices = np.array(range(len(self.sample_idx_train)))
        self.n_valid_samples = len(self.sample_idx_valid)
        self.n_train_samples = len(self.sample_idx_train)

    def define_validation_set_holdout_pop(self, holdout_pop, superpopulations_file):

        print(superpopulations_file)

        superpops = pd.read_csv(superpopulations_file, header=None).to_numpy()
        print(superpops)

        a = pd.read_csv(self.filebase + ".fam", header=None)

        b = a.to_numpy()
        c = [b[i][0].split() for i in range(len(b))]
        fam = np.array(c)

        pops = fam[:, 0]
        print(pops)
        print(superpops[:, 0])

        # This line below here takes an array of the population for each sample (pops) and sends it to a new array with the corresponding superpopulation.
        # Thus, given pop -> respective superpop. This is done in one line, and maybe not too understandable.
        sample_superpop = np.array([superpops[np.where(pops[i] == superpops[:, 0])[0][0], 1] for i in range(len(pops))])

        print(sample_superpop)
        print(holdout_pop)

        # The indices of all samples
        idx = np.arange(len(pops))

        # Take as validation samples the indices that corresponds to the held-out population.
        self.sample_idx_valid = np.where(sample_superpop == holdout_pop)[0]

        # Train on the rest.
        self.sample_idx_train = np.delete(idx, self.sample_idx_valid)

        print(self.sample_idx_valid)
        print(self.sample_idx_train)

        self.train_set_indices = np.array(range(len(self.sample_idx_train)))
        self.n_valid_samples = len(self.sample_idx_valid)
        self.n_train_samples = len(self.sample_idx_train)

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

        _, _, self.sample_idx_train, self.sample_idx_valid = get_test_samples_stratified(self.genotypes_train_orig,
                                                                                         self.ind_pop_list_train_orig,
                                                                                         validation_split)

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
        input_data_valid = np.full((len(self.sample_idx_valid), self.genotypes_train_orig.shape[1], 2), 1.0,
                                   dtype=np.dtype('f4'))

        genotypes_valid = np.copy(self.genotypes_train_orig[self.sample_idx_valid])

        mask_valid = np.full(input_data_valid[:, :, 0].shape, 1)

        if not self.impute_missing:
            # set the ones that are originally missing to 0 in mask (sinc we havent imputed them with RR)
            mask_valid[np.where(genotypes_valid == self.missing_val)] = 0

        if sparsify > 0.0:
            self._sparsify(mask_valid, 1.0 - sparsify)

            missing_idx_valid = np.where(mask_valid == 0)

            genotypes_valid[missing_idx_valid] = self.missing_val

        input_data_valid[:, :, 0] = genotypes_valid
        input_data_valid[:, :, 1] = mask_valid

        targets = self.genotypes_train_orig[self.sample_idx_valid]

        return input_data_valid, targets, self.ind_pop_list_train_orig[self.sample_idx_valid]

    def reset_batch_index(self):
        '''
        Reset the internal sample counter for batches. So the next batch will start with sample 0.

        '''
        self.train_batch_location = 0

    # This is perhaps not used in new input pipeline, trash?
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
            idx = list(range(self.train_batch_location, len(self.sample_idx_train), 1)) + list(
                range(0, n_samples - (len(self.sample_idx_train) - self.train_batch_location), 1))
            self.train_batch_location = n_samples - (len(self.sample_idx_train) - self.train_batch_location)

        return self.train_set_indices[idx]

    def get_train_batch(self, sparsify, n_samples_batch):
        '''
        Get n_samples_batch train samples, with genotypes randomly set to missing with probability sparsify.
        Fetch n_samples sequentially starting at index self.train_batch_location, looping over the current train set

        If validation set has been defined, return train samples exluding the validation samples.

        :param sparsify:
        :param n_samples_batch: number of samples in batch
        :return: input_data_train_batch (n_samples x n_markers x 2): sparsified  genotypes with mask specifying missing values of trai batch.
                                                                               originally missing + removed by sparsify are indicated by value 0
                 target_data_train_batch (n_samples x n_markers): original genotypes of this train batch
                 ind_pop_list_train_batch (n_samples x 2) : individual and population IDs of train batch samples

        '''
        indices_this_batch = self._get_indices_looped(n_samples_batch)
        genotypes_train = self.genotypes_train_orig[self.sample_idx_train[indices_this_batch]]
        genotypes_train_orig = genotypes_train
        if not self.impute_missing:
            # set the ones that are originally missing to 0 in mask (sinc we havent imputed them with RR)
            mask_train = np.where(genotypes_train == self.missing_val)
        else:
            mask_train = np.full((n_samples_batch, genotypes_train.shape[1]), 1, dtype=np.bool_)

        if sparsify > 0.0:
            self._sparsify(mask_train, 1.0 - sparsify)
            # fill genotypes with original valid genotypes and sparsify according to binary_mask_train
            genotypes_train = np.where(mask_train == 0, self.missing_val, genotypes_train)

        (input_data_train, targets) = helper_get_train_batch(sparsify, n_samples_batch, genotypes_train,
                                                             genotypes_train_orig, mask_train, indices_this_batch,
                                                             self.missing_val)

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
        input_data_train = np.full((self.n_train_samples, self.genotypes_train_orig[self.sample_idx_train].shape[1], 2),
                                   1.0)

        genotypes_train = np.copy(self.genotypes_train_orig[self.sample_idx_train])

        # the mask is all ones
        mask_train = np.full(input_data_train[:, :, 0].shape, 1)

        if not self.impute_missing:
            # set the ones that are originally missing to 0 in mask (sinc we havent imputed them with RR)
            mask_train[np.where(genotypes_train == self.missing_val)] = 0

        if sparsify > 0.0:
            self._sparsify(mask_train, 1.0 - sparsify)

            # indices of originally missing data + artifically sparsified data
            missing_idx_train = np.where(mask_train == 0)

            # fill genotypes with original valid genotypes and sparsify according to binary_mask_train
            genotypes_train[missing_idx_train] = self.missing_val

        input_data_train[:, :, 0] = genotypes_train
        input_data_train[:, :, 1] = mask_train

        print("In DG.get_train_set: number of " + str(self.missing_val) + " genotypes in train: " + str(
            len(np.where(input_data_train[:, :, 0] == self.missing_val)[0])))
        print("In DG.get_train_set: number of -9 genotypes in train: " + str(
            len(np.where(input_data_train[:, :, 0] == -9)[0])))
        print("In DG.get_train_set: number of 0 values in train mask: " + str(
            len(np.where(input_data_train[:, :, 1] == 0)[0])))
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

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p, bruteforce=True) >= 0


def convex_hull_error(coords_by_pop, plot=False, min_points_required=3):
    '''

    Calculate the hull error of projected coordinates of the populations defined by coords_by_pop

    For every population: calculate the fraction that other population's points make up of all the points inside THIS population's convex hull
    Return the average of this over populations.

    :param coords_by_pop: dict mapping population to list of coordinates (n_samples x n_dim)
    :param min_points_required: min number of points needed to define a simplex for the convex hull.

    :return: the hull error
    '''
    all_pop_coords = functools.reduce(lambda a, b: a + b, [coords_by_pop[pop] for pop in coords_by_pop.keys()])

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

            assert len(other_pop_coords) + num_points_in_pop == len(all_pop_coords), "Number of inds does not match up!"

            if num_points_in_pop >= min_points_required:
                num_other_points_in_hull = sum(in_hull(other_pop_coords, this_pop_coords))
                frac_other_pop_points_in_hull = num_other_points_in_hull / (
                            num_points_in_pop + num_other_points_in_hull)
                frac_sum += frac_other_pop_points_in_hull

                if plot:
                    hull = ConvexHull(this_pop_coords)
                    plt.scatter(this_pop_coords[:, 0], this_pop_coords[:, 1], color="red")
                    plt.scatter(other_pop_coords[:, 0], other_pop_coords[:, 1], color="black", alpha=0.8)
                    for simplex in hull.simplices:
                        plt.plot(this_pop_coords[simplex, 0], this_pop_coords[simplex, 1], 'k-')
                    plt.show()
                    plt.close()

            else:
                print("Too few for hull: " + str(pop))
                if num_points_in_pop < 3:
                    print("--  shape: " + str(this_pop_coords.shape) + ": skipping")
                    continue
                elif num_points_in_pop == 3:
                    n_dim = int(num_points_in_pop) - 1
                else:
                    n_dim = int(num_points_in_pop) - 2

                print("--  shape: " + str(this_pop_coords.shape) + ": using " + str(n_dim) + " dimensions instead")

                num_other_points_in_hull = sum(in_hull(other_pop_coords[:, 0:n_dim], this_pop_coords[:, 0:n_dim]))
                frac_other_pop_points_in_hull = num_other_points_in_hull / (
                            num_points_in_pop + num_other_points_in_hull)
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
    genotype_conc_per_marker = counts / n_samples
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


def f1_score_kNN(x, labels, labels_to_use, k=5):
    if k > 0:
        classifier = KNeighborsClassifier(n_neighbors=k)
    elif k == 0:
        classifier = NearestCentroid()
    elif k == -1:
        classifier = QuadraticDiscriminantAnalysis()
    elif k == -2:
        classifier = GaussianNB()
    elif k == -3:
        classifier = LinearDiscriminantAnalysis()
    if k > 0:
        classifier.fit(x, labels)
    else:  # Never used
        classifier.fit(np.concatenate((x, x + (np.random.random_sample(x.shape) - 0.5) * 1e-4)),
                       np.concatenate((labels, labels)))
    predicted_labels = classifier.predict(x)
    # this returns a vector of f1 scores per population
    f1_score_per_pop = f1_score(y_true=labels, y_pred=predicted_labels, labels=labels_to_use, average=None)
    f1_score_avg = f1_score(y_true=labels, y_pred=predicted_labels, average="micro")

    return f1_score_avg, f1_score_per_pop


def my_tf_round(x, d=2, base=0.5):
    '''
    Round input to nearest multiple of base, considering d decimals.

    :param x: tensor
    :param d: number of decimals to consider
    :param base: rounding to nearest base
    :return: x rounded
    '''
    multiplier = tf.constant(10 ** d, dtype=x.dtype)
    x = base * tf.math.round(tf.math.divide(x, base))
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
        genos[m] = np.add(np.multiply(genos[m], stds[m]), means[m])

    output = tf.map_fn(lambda x: my_tf_round(x, base=1.0), genos.T)

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
        _ = self.accruary_metric.update_state(y_true=y_true,
                                              y_pred=y_pred)  # self.accruary_metric.update_state(y_true=y_true*2, y_pred = tf.one_hot(tf.cast(y_pred*2+1e-3, tf.int32), 3))
        return y_pred

    def result(self):
        return self.accruary_metric.result()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.accruary_metric.reset_states()


def write_h5(filename, dataname, data, replace_file=False):
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
                hf.create_dataset(dataname, data=data)
            except (RuntimeError, ValueError):
                print("Replacing dataset {0} in {1}".format(dataname, filename))
                del hf[dataname]
                hf.create_dataset(dataname, data=data)
    except OSError:
        print("Could not write to file {0}.".format(filename))
        if replace_file:
            print("Replacing {0}.".format(filename))
            with h5py.File(filename, 'w') as hf:
                try:
                    hf.create_dataset(dataname, data=data)
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

    pop_superpop_list = np.genfromtxt(file, usecols=(0, 1), dtype=str, delimiter=",")
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
        ind_pop_map_list = np.genfromtxt(mapfile, usecols=(0, 2), dtype=str)
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
        ind_pop_list = np.genfromtxt(filestart + ".fam", usecols=(1, 0), dtype=str)
        print("Reading ind pop list from " + filestart + ".fam")
    except:
        ind_pop_list = np.genfromtxt(filestart + ".ind", usecols=(0, 2), dtype=str)
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


def get_coords_by_pop(filestart_fam, coords, pop_subset=None, ind_pop_list=[]):
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
        unique_pops = np.unique(ind_pop_list[:, 1])

    else:
        ind_pop_list = get_ind_pop_list(filestart_fam)
        unique_pops = get_unique_pop_list(filestart_fam)

    pop_list = ind_pop_list[:, 1]

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
    for i in os.listdir(train_directory + "/weights"):
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

    return epochs_combined, values_combined


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
        plt.title(", ".join(["{:.2f} : {}".format(u, c) for (u, c) in d]), fontdict={'fontsize': 9})

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
    superpops = np.unique(pop_superpop_list[:, 1])

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
    pop_list = np.array(ind_pop_list[:, 1])
    panel_pops = np.unique(pop_list)
    n_samples = len(pop_list)

    if test_split == 0.0:
        return range(len(ind_pop_list)), []

    ################### stratify cant handle classes with only one sample ########################

    else:
        counter_dict = {}
        for pop in panel_pops:
            counter_dict[pop] = 0

        for ind, pop in ind_pop_list:
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
            print(str(test_split * n_samples) + "{0} is too few samples for " + str(
                n_pops) + " classes. Setting split fraction to " + str(test_split))

        sample_idx = range(len(pop_list))
        sample_idx_train, sample_idx_test, pops_train, pops_test = train_test_split(sample_idx, pop_list,
                                                                                    test_size=test_split,
                                                                                    stratify=pop_list, random_state=1)
        sample_idx_train = np.array(sample_idx_train)
        sample_idx_test = np.array(sample_idx_test)

        pops_train_recon = pop_list[sample_idx_train]
        pops_test_recon = pop_list[sample_idx_test]

        for i in range(len(pops_train_recon)):
            assert pops_train_recon[i] == pops_train[i]
        for i in range(len(pops_test_recon)):
            assert pops_test_recon[i] == pops_test[i]

        return sample_idx_train, sample_idx_test


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
    pop_list = np.array(ind_pop_list[:, 1])
    panel_pops = np.unique(pop_list)
    n_samples = len(pop_list)

    if test_split == 0.0:
        return genotypes, [], range(len(ind_pop_list)), []

    ################### stratify cant handle classes with only one sample ########################

    else:
        counter_dict = {}
        for pop in panel_pops:
            counter_dict[pop] = 0

        for ind, pop in ind_pop_list:
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
            print(str(test_split * n_samples) + "{0} is too few samples for " + str(
                n_pops) + " classes. Setting split fraction to " + str(test_split))

        sample_idx = range(len(genotypes))
        genotypes_train, genotypes_test, sample_idx_train, sample_idx_test, pops_train, pops_test = train_test_split(
            genotypes, sample_idx, pop_list, test_size=test_split, stratify=pop_list, random_state=1)
        sample_idx_train = np.array(sample_idx_train)
        sample_idx_test = np.array(sample_idx_test)

        pops_train_recon = pop_list[sample_idx_train]
        pops_test_recon = pop_list[sample_idx_test]

        for i in range(len(pops_train_recon)):
            assert pops_train_recon[i] == pops_train[i]
        for i in range(len(pops_test_recon)):
            assert pops_test_recon[i] == pops_test[i]

        return genotypes_train, genotypes_test, sample_idx_train, sample_idx_test


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
                 batch_size, normalization_mode="genotypewise01", recombination_rate=0.0,
                 normalization_options={"flip": False, "missing_val": -1.0, "num_markers": None}, impute_missing=True,
                 sparsifies=False):

        self.filebase = filebase
        self.batch_size = batch_size
        self._define_samples()
        self.normalization_mode = normalization_mode
        self.normalization_options = normalization_options
        self.impute_missing = impute_missing
        self.sparsifies = sparsifies
        self.missing_val = normalization_options["missing_val"]
        self.recombination_rate = recombination_rate
        self.marker_count()  # This sets the number of markers
        if normalization_options["num_markers"] is not None:
            self.n_markers = normalization_options["num_markers"]
        if sparsifies == False:
            self.sparsify_input = False
        else:
            self.sparsify_input = True
        if recombination_rate == 0.0:
            self.total_batch_size = batch_size
            self.recombination = False

        else:
            self.recombination = True
            self.total_batch_size = batch_size + np.int(np.ceil(self.recombination_rate * self.batch_size))
            # Initialize some snp-related data, such as the chromosomes and the centimorgan distances
            self.get_chromosome_data()

    def generator_recombine_here(self, pref_chunk_size, training, shuffle=True):

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
            recomb_rate = self.recombination_rate
            # This yields randomized chunks and batches for each epoch. Essentially shuffling the samples
            if shuffle:

                indices = tf.random.shuffle(self.sample_idx_train)

            else:
                indices = self.sample_idx_train[np.arange(0, n_samples)]


        elif not training:
            training = False
            n_samples = self.n_valid_samples
            recomb_rate = 0
            if shuffle:
                indices = tf.random.shuffle(self.sample_idx_valid)

            else:
                indices = self.sample_idx_valid[np.arange(0, n_samples)]

        recombined_samples_per_batch = np.int(
            np.ceil(recomb_rate * self.batch_size))  # fixed number of samples that we just append
        self.recombined_batch_size = self.batch_size + recombined_samples_per_batch

        chunk_size = pref_chunk_size // self.batch_size * self.batch_size  # Make the preferred chunk size an integer multiple of the batch size
        num_chunks = np.ceil(n_samples / chunk_size)
        chunks_read = 0
        indices = tf.cast(indices, tf.int32)
        while chunks_read < num_chunks:

            chunk_indices = indices[chunk_size * chunks_read: chunk_size * (chunks_read + 1)]

            batches_per_chunk = np.ceil(len(chunk_indices) / self.batch_size)  # Number of
            if self.recombination and not chunks_read == (num_chunks - 1):
                recombined_samples_this_chunk = np.int(recombined_samples_per_batch * batches_per_chunk)

                parent_inds = tf.gather(indices,
                                        indices=tf.random.uniform((2 * recombined_samples_this_chunk,), minval=0,
                                                                  maxval=n_samples, dtype=tf.dtypes.int32))

                chunk_indices = tf.concat([chunk_indices, parent_inds], axis=0)
            else:
                parent_inds = 0

            un, idx = tf.unique(chunk_indices)

            df = pd.read_parquet(self.filebase + ".parquet", columns=tf.strings.as_string(un).numpy())
            # Read only the selected indices into memory,
            # this read is made to be as large as possible, consisting of several batches, but (perhaps not the entire dataset)

            df_numpy = pd.DataFrame.to_numpy(df)[0:self.n_markers, list(idx)]

            # If on the last batch, don't add any recombined samples
            if not chunks_read == (num_chunks - 1):
                original_samples = df_numpy[:, 0:int(batches_per_chunk * self.batch_size)]
                parent_samples = df_numpy[:, int(batches_per_chunk * self.batch_size):]
            else:
                original_samples = df_numpy
                parent_samples = []

            # Here, batches per chunk will be the same for all chunks, except for the last, which will most likely be a bit smaller
            # batches_per_chunk = np.ceil(len(chunk_indices) / self.batch_size)

            # Set nan values to 9. For e.g. the MAGIC wheat dataset, that has missing values corresponding represented by NaN
            df_numpy[np.where(np.isnan(df_numpy))] = 9
            batches_read = 0

            while batches_read < batches_per_chunk:

                genotypes_train = original_samples[:,
                                  self.batch_size * batches_read: self.batch_size * (batches_read + 1)]
                # parent_samples[:,]
                inds = chunk_indices[self.batch_size * batches_read: self.batch_size * (batches_read + 1)]
                t_rec = time.perf_counter()
                if self.recombination and not chunks_read == (num_chunks - 1):

                    parent_genotypes = parent_samples[:,
                                       batches_read * recombined_samples_per_batch * 2: batches_read * recombined_samples_per_batch * 2 + 2 * recombined_samples_per_batch]
                    p_inds = parent_inds[
                             batches_read * recombined_samples_per_batch * 2: batches_read * recombined_samples_per_batch * 2 + 2 * recombined_samples_per_batch]

                    offspring_data = np.zeros(shape=(self.n_markers, recombined_samples_per_batch))

                    for sample_number in range(recombined_samples_per_batch):

                        u = tf.random.uniform(shape=self.recomb_prob.shape, minval=0, maxval=1, )

                        cut_idx = tf.where(u < self.recomb_prob)
                        cut_idx = tf.concat([tf.zeros([1, 1], dtype=tf.int64), cut_idx], axis=0)
                        cut_idx = tf.concat(
                            [cut_idx, tf.ones([1, 1], dtype=tf.int64) * tf.cast(self.n_markers, tf.int64)], axis=0)

                        b = [tf.range(cut_idx[i], cut_idx[i + 1]) for i in range(tf.shape(cut_idx)[0] - 1)]

                        p1 = b[::2]
                        p2 = b[1::2]
                        p1_inds = np.array([])
                        p2_inds = np.array([])

                        for i in p1:
                            p1_inds = np.append(p1_inds, i)
                        for i in p2:
                            p2_inds = np.append(p2_inds, i)

                        p1_inds = p1_inds.astype(int)
                        p2_inds = p2_inds.astype(int)

                        parent_array = parent_genotypes[:, 2 * sample_number: 2 * sample_number + 2]
                        offspring_data[p1_inds, sample_number] = parent_array[p1_inds, 0]
                        offspring_data[p2_inds, sample_number] = parent_array[p2_inds, 1]

                    genotypes_train = np.append(genotypes_train, offspring_data, axis=1)
                    inds = np.append(inds, p_inds)

                # If the shape does not match the usual batch size, i.e., we are on the last batch, also yield the keyword "last_batch" as a True bool. Need to explicitly give this since
                # Tensorflow does not know the actual batch size at run time, and I need to create the mask not having access to .shape commands at run time.

                if genotypes_train.shape[1] == (self.batch_size + 1 * recombined_samples_per_batch):
                    yield genotypes_train, self.ind_pop_list_train_orig[inds, :], [False, training]
                else:
                    yield genotypes_train, self.ind_pop_list_train_orig[inds, :], [True, training]

                batches_read += 1
            chunks_read += 1

    def generator(self, pref_chunk_size, training, shuffle=True):

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
        # TODO: There seems to be an issue where (in the very rare case) that when n_valid_batches % batch_size = 0, it sends a last batch with 0 samples, which is not good. Fix this sometime, meanwhile, just choose another set of parameters.

        if training:
            training = True
            n_samples = self.n_train_samples
            recomb_rate = self.recombination_rate
            # This yields randomized chunks and batches for each epoch. Essentially shuffling the samples
            if shuffle:
                indices = tf.random.shuffle(self.sample_idx_train)
            else:
                indices = self.sample_idx_train[np.arange(0, n_samples)]

        elif not training:
            training = False
            n_samples = self.n_valid_samples
            recomb_rate = 0
            if shuffle:
                indices = tf.random.shuffle(self.sample_idx_valid)
            else:
                indices = self.sample_idx_valid[np.arange(0, n_samples)]

        recombined_samples_per_batch = np.int(
            np.ceil(recomb_rate * self.batch_size))  # fixed number of samples that we just append
        self.recombined_batch_size = self.batch_size + recombined_samples_per_batch

        chunk_size = pref_chunk_size // self.batch_size * self.batch_size  # Make the preferred chunk size an integer multiple of the batch size
        num_chunks = np.ceil(n_samples / chunk_size)
        chunks_read = 0
        indices = tf.cast(indices, tf.int32)
        while chunks_read < num_chunks:

            chunk_indices = indices[chunk_size * chunks_read: chunk_size * (chunks_read + 1)]
            batches_per_chunk = np.ceil(len(chunk_indices) / self.batch_size)
            if self.recombination and training:
                recombined_samples_this_chunk = np.int(recombined_samples_per_batch * batches_per_chunk)

                parent_inds = tf.gather(indices,
                                        indices=tf.random.uniform((2 * recombined_samples_this_chunk,), minval=0,
                                                                  maxval=n_samples, dtype=tf.dtypes.int32))

                chunk_indices = tf.concat([chunk_indices, parent_inds], axis=0)

            un, idx = tf.unique(chunk_indices)
            df = pd.read_parquet(self.filebase + ".parquet", columns=tf.strings.as_string(un).numpy())
            # Read only the selected indices into memory,
            # this read is made to be as large as possible, consisting of several batches, but (perhaps not the entire dataset)

            df_numpy = pd.DataFrame.to_numpy(df)[0:self.n_markers, list(idx)]

            # If on the last batch, don't add any recombined samples
            if not chunks_read == (num_chunks - 1):

                original_samples = df_numpy[:, 0:int(batches_per_chunk * self.batch_size)]
                parent_samples = df_numpy[:, int(batches_per_chunk * self.batch_size):]

            else:

                if training:
                    if self.n_train_samples % chunk_size == 0:
                        num_samples_last_chunk = self.n_train_samples
                    else:
                        num_samples_last_chunk = self.n_train_samples % chunk_size

                elif not training:
                    if self.n_valid_samples % chunk_size == 0:
                        num_samples_last_chunk = self.n_valid_samples
                    else:
                        num_samples_last_chunk = self.n_valid_samples % chunk_size

                original_samples = df_numpy[:, 0:int(num_samples_last_chunk)]

                parent_samples = df_numpy[:, int(num_samples_last_chunk):]

            # Here, batches per chunk will be the same for all chunks, except for the last, which will most likely be a bit smaller
            # batches_per_chunk = np.ceil(len(chunk_indices) / self.batch_size)

            # Set nan values to 9. For e.g. the MAGIC wheat dataset, that has missing values corresponding represented by NaN
            df_numpy[np.where(np.isnan(df_numpy))] = 9
            batches_read = 0

            while batches_read < batches_per_chunk:

                genotypes_train = original_samples[:,
                                  self.batch_size * batches_read: self.batch_size * (batches_read + 1)]

                inds = chunk_indices[self.batch_size * batches_read: self.batch_size * (batches_read + 1)]

                if self.recombination and training:

                    parent_genotypes = parent_samples[:,
                                       batches_read * recombined_samples_per_batch * 2: batches_read * recombined_samples_per_batch * 2 + 2 * recombined_samples_per_batch]
                    p_inds = parent_inds[
                             batches_read * recombined_samples_per_batch * 2: batches_read * recombined_samples_per_batch * 2 + 2 * recombined_samples_per_batch]

                    genotypes_train = np.append(genotypes_train, parent_genotypes, axis=1)
                    inds = np.append(inds, p_inds)
                    if (genotypes_train.shape[1] == (self.batch_size + 2 * recombined_samples_per_batch)):
                        yield genotypes_train, self.ind_pop_list_train_orig[inds, :], [False, training]
                    else:
                        yield genotypes_train, self.ind_pop_list_train_orig[inds, :], [True, training]

                else:

                    # If the shape does not match the usual batch size, i.e., we are on the last batch, also yield the keyword "last_batch" as a True bool. Need to explicitly give this since
                    # Tensorflow does not know the actual batch size at run time, and I need to create the mask not having access to .shape commands at run time.

                    if genotypes_train.shape[1] == self.batch_size:
                        yield genotypes_train, self.ind_pop_list_train_orig[inds, :], [False, training]
                    else:
                        yield genotypes_train, self.ind_pop_list_train_orig[inds, :], [True, training]

                batches_read += 1
            chunks_read += 1

    def recombination_mapping(self, x, inds, args):
        """
            x here is one batch of data, which has shape (n_markers, n_samples+n_parents )

            The input to this function includes the genotypes for batch_size original samples,
            and also a number or parent samples that will be used in the recombination.

            the output will then have the shape (n_markers, n_samples+n_parents/2)

            This severely needs an overhaul, and improved performance




        """

        genotypes_train = x
        last_batch = args[0]  # This is false for all batches except the last. It is needed to control the sample size
        training = args[1]  # This is true if the dataset is generated for training (not validation)

        recombined_samples_per_batch = int(
            self.recombination_rate * self.batch_size)  # fixed number of samples that we just append to the end of the output

        num_samples = self.n_train_samples_last_batch * tf.cast(last_batch, tf.int32) + self.batch_size * (
                1 - tf.cast(last_batch, tf.int32))  # The number of samples, will be different if on last batch

        # Extraxt the original training samples, and the parent samples to be used in recombination
        genotypes_orig = genotypes_train[:, 0:num_samples]
        genotypes_parents = genotypes_train[:, num_samples:]
        offspring_tensor = tf.zeros([self.n_markers, recombined_samples_per_batch])
        t0 = time.perf_counter()
        for n in range(recombined_samples_per_batch):
            # Create new sample
            offspring = tf.reshape(self.create_offspring(tf.gather(genotypes_parents, [2 * n, 2 * n + 1], axis=1)),
                                   [self.n_markers, ])
            if n == 0:
                listoid2 = tf.reshape(offspring, [self.n_markers, 1])
            else:
                listoid2 = tf.concat([listoid2, tf.reshape(offspring, [self.n_markers, 1])], axis=1)

        offspring_tensor = listoid2
        # tf.print(tf.shape(offspring_tensor))
        # tf.print(time.perf_counter()-t0)
        # t1 = time.perf_counter()

        offspring_tensor = self.create_offspring_unlooped(genotypes_parents)

        # t#f.print(tf.shape(offspring_tensor))
        # t#f.print(time.perf_counter()-t1)

        # exit()
        # Append the new samples to the training set.

        genotypes_train = tf.concat([genotypes_orig, offspring_tensor], axis=1)

        return genotypes_train, inds, [tf.cast(last_batch, tf.bool), training]

    def create_offspring_unlooped(self, parent_array):

        # Using this function we create all offspring at once, instead of creating them one at a time
        # TODO: Verfify the implementation, e.g., check that we maintain arrpoixate means, and that we dont get any weird values.

        recomb_prob = tf.cast(self.recomb_prob, tf.float32)
        recomb_prob2 = tf.tile(tf.reshape(recomb_prob, [self.n_markers - 1, 1]),
                               [1, tf.cast(tf.shape(parent_array)[1] / 2, tf.int32)])

        u = tf.random.uniform(shape=tf.shape(recomb_prob2), minval=0, maxval=1)

 
        a1 = tf.math.cumsum(tf.cast(u < recomb_prob2, tf.int32), axis=0) % 2
        a = tf.concat([a1, a1[tf.newaxis, -1, :]], axis=0)
        ind0 = tf.where(a == 0)
        ind1 = tf.where(a == 1)

        ind_0 = tf.stack([ind0[:, 0], ind0[:, 1]], axis=1)  # tf.zeros(tf.shape(ind0[:, 0]), tf.int64)], axis=1)
        ind_1 = tf.stack([ind1[:, 0], ind1[:, 1] + tf.cast(tf.shape(parent_array)[1] / 2, tf.int64)],
                         axis=1)  # tf.ones(tf.shape(ind1[:, 0]), tf.int64)], axis=1)

        parent_0_vals = tf.gather_nd(parent_array, ind_0)  # parent_array[ind0, 0]
        parent_1_vals = tf.gather_nd(parent_array, ind_1)  # parent_array[ind1, 1]

        st0 = tf.scatter_nd(ind0, parent_0_vals,
                            shape=[self.n_markers, tf.cast(tf.shape(parent_array)[1] / 2, dtype=tf.int64)])
        st1 = tf.scatter_nd(ind1, parent_1_vals,
                            shape=[self.n_markers, tf.cast(tf.shape(parent_array)[1] / 2, dtype=tf.int64)])

        offspring = st0 + st1
        return offspring

    def recombination_mapping2(self, x, inds, args):
        """
            x here is one batch of data, which has shape (n_markers, n_samples+n_parents )

            The input to this function includes the genotypes for batch_size original samples,
            and also a number or parent samples that will be used in the recombination.

            the output will then have the shape (n_markers, n_samples+n_parents/2)

            This severely needs an overhaul, and improved performance


        """

        genotypes_train = x
        last_batch = args[0]  # This is false for all batches except the last. It is needed to control the sample size
        training = args[1]  # This is true if the dataset is generated for training (not validation)

        recombined_samples_per_batch = int(
            self.recombination_rate * self.batch_size)  # fixed number of samples that we just append to the end of the output

        num_samples = self.n_train_samples_last_batch * tf.cast(last_batch, tf.int32) + self.batch_size * (
                    1 - tf.cast(last_batch, tf.int32))  # The number of samples, will be different if on last batch

        # Extraxt the original training samples, and the parent samples to be used in recombination
        genotypes_orig = genotypes_train[:, 0:num_samples]
        genotypes_parents = genotypes_train[:, num_samples:]

        offspring_tensor = tf.zeros([self.n_markers, recombined_samples_per_batch])

        for n in range(recombined_samples_per_batch):
            # Create new sample
            offspring = tf.reshape(self.create_offspring(tf.gather(genotypes_parents, [2 * n, 2 * n + 1], axis=1)),
                                   [self.n_markers, ])

            # Create a sparse matrix with this sample in the correct position, some other tested ways were unsuccessful.
            temp_inds = tf.concat([tf.reshape(tf.range(self.n_markers, dtype=tf.int32), [self.n_markers, 1]),
                                   n * tf.ones([self.n_markers, 1], dtype=tf.int32)], axis=1)

            st_temp = tf.sparse.SparseTensor(indices=tf.cast(temp_inds, tf.int64),
                                             values=tf.cast(offspring, tf.float32),
                                             dense_shape=[self.n_markers, recombined_samples_per_batch])
            # Add the sparse matrix to the list of new samples
            offspring_tensor = tf.sparse.add(offspring_tensor, st_temp)

        # Append the new samples to the training set.
        genotypes_train = tf.concat([genotypes_orig, offspring_tensor], axis=1)

        return genotypes_train, inds, [tf.cast(last_batch, tf.bool), training]

    def get_chromosome_data(self):

        if os.path.isfile(self.filebase + ".snp"):
            snp_file = self.filebase + ".snp"

            snp_data = np.genfromtxt(snp_file, usecols=(0, 1, 2, 3, 4, 5), dtype=str)[0:self.n_markers]

            self.chrs = snp_data[:, 1].astype(int)
            cm_dist = snp_data[:, 2].astype(float)

        elif os.path.isfile(self.filebase + ".bim"):
            bim_file = self.filebase + ".bim"

            snp_data = np.genfromtxt(bim_file, usecols=(0, 1, 2, 3, 4, 5), dtype=str)[0:self.n_markers]
            self.chrs = snp_data[:, 0].astype(int)

            if np.all(snp_data[:, 2].astype(int) == 0):
                bp_per_cm = 1000000
                print("No cM distances found, estimating that 1cm = 1e6 bp")

                cm_dist = snp_data[:, 3].astype(int) / bp_per_cm / 100

            else:
                cm_dist = snp_data[:, 2].astype(float)

        else:
            print("Can't recombine, no available data to approximate genotype map.")
            exit()

        extra_prob = 1
        self.recomb_prob = 1 / 2 * (1 - np.exp(-4 * (cm_dist[1:] - cm_dist[:-1]))) * extra_prob

        tf.print(tf.reduce_max(self.recomb_prob))

    def create_offspring(self, parent_array):

        recomb_prob = self.recomb_prob

        u = tf.random.uniform(shape=recomb_prob.shape, minval=0, maxval=1)
        cut_idx = (tf.where(u < recomb_prob))
        b = tf.range(self.n_markers, dtype=tf.int64)
        a = tf.reduce_sum(tf.cast(b[:] < cut_idx, tf.int32), axis=0) % 2

        ind0 = tf.where(a == 0)
        ind1 = tf.where(a == 1)

        ind_0 = tf.stack([ind0[:, 0], tf.zeros(tf.shape(ind0[:, 0]), tf.int64)], axis=1)
        ind_1 = tf.stack([ind1[:, 0], tf.ones(tf.shape(ind1[:, 0]), tf.int64)], axis=1)

        parent_0_vals = tf.gather_nd(parent_array, ind_0)  # parent_array[ind0, 0]
        parent_1_vals = tf.gather_nd(tf.reverse(parent_array, axis=[0]), ind_1)  # parent_array[ind0, 0]

        st0 = tf.scatter_nd(ind0, parent_0_vals, tf.constant([self.n_markers], dtype=tf.int64))
        st1 = tf.scatter_nd(ind1, parent_1_vals, tf.constant([self.n_markers], dtype=tf.int64))

        offspring = st0 + st1
        return offspring

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
            if training:
                num_samples = self.total_batch_size
            else:
                num_samples = self.batch_size
        else:

            if training:
                num_samples = self.n_train_samples_last_batch + self.recombination * (
                            self.total_batch_size - self.batch_size)
            else:
                num_samples = self.n_valid_samples_last_batch

        # a is a sparse tensor containing the missing values.
        a = tf.sparse.SparseTensor(indices=missing_indices,
                                   values=tf.ones(shape=tf.shape(missing_indices)[0], dtype=tf.float32),
                                   dense_shape=(num_samples, self.n_markers))

        indices = missing_indices[:, 1]
        x = tf.transpose(x)

        if self.impute_missing:
            b = tf.sparse.SparseTensor(indices=missing_indices,
                                       values=(tf.gather(self.most_common_genos, indices=indices) - 9),
                                       dense_shape=(num_samples, self.n_markers))

            x = tf.sparse.add(x, b)

        if self.normalization_mode == "genotypewise01":
            if self.normalization_options["flip"]:
                # This operation results in the missing values having a value of -7/2, wrong! Amend this by adding 3.5-missing_val
                x = -(x - 2) / 2
                if not self.impute_missing:
                    x = tf.sparse.add(x, a * (3.5 - self.missing_val))
            else:
                # This operation results in the missing values having a value of 4.5, wrong! Amend this by subtracting 4.5-missing_val
                x = x / 2
                if not self.impute_missing:
                    x = tf.sparse.add(x, a * (self.missing_val - 4.5))

        elif self.normalization_mode == "standard" or self.normalization_mode == "smartPCAstyle":
            if self.normalization_options["flip"]:
                x = -(x - 2)
            x2 = (x - self.scaler.mean_) / tf.math.sqrt(self.scaler.var_)
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
            sparsify_fraction = tf.random.shuffle(self.sparsifies)[0]
        else:
            sparsify_fraction = 0.0
        try:
            if self.project:
                sparsify_fraction = 0.0
        except:
            pass

        if self.training:
            num_samples = self.total_batch_size
        else:
            num_samples = self.batch_size
        missing_value = self.missing_val

        if last_batch:

            if self.training:
                mask = tf.experimental.numpy.full(shape=(
                self.n_train_samples_last_batch + self.recombination * (self.total_batch_size - self.batch_size),
                self.n_markers), fill_value=1.0,
                                                  dtype=tf.float32)
                b = tf.random.uniform(shape=(
                self.n_train_samples_last_batch + self.recombination * (self.total_batch_size - self.batch_size),
                self.n_markers), minval=0, maxval=1)

                indices = tf.where(b < sparsify_fraction)

                b = tf.sparse.SparseTensor(indices=indices,
                                           values=(tf.repeat(-1.0, tf.shape(indices)[0])),
                                           dense_shape=(self.n_train_samples_last_batch + self.recombination * (
                                                       self.total_batch_size - self.batch_size), self.n_markers))
                mask = tf.sparse.add(mask, b)

            else:
                mask = tf.experimental.numpy.full(shape=(self.n_valid_samples_last_batch, self.n_markers),
                                                  fill_value=1.0,
                                                  dtype=tf.float32)
                b = tf.random.uniform(shape=(self.n_valid_samples_last_batch, self.n_markers), minval=0, maxval=1)

                indices = tf.where(b < sparsify_fraction)

                b = tf.sparse.SparseTensor(indices=indices,
                                           values=(tf.repeat(-1.0, tf.shape(indices)[0])),
                                           dense_shape=(self.n_valid_samples_last_batch, self.n_markers))
                mask = tf.sparse.add(mask, b)

        else:

            mask = tf.experimental.numpy.full(shape=(num_samples, self.n_markers), fill_value=1.0,
                                              dtype=tf.float32)
            b = tf.random.uniform(shape=(num_samples, self.n_markers), minval=0, maxval=1)
            indices = tf.where(b < sparsify_fraction)
            b = tf.sparse.SparseTensor(indices=indices,
                                       values=(tf.repeat(-1.0, tf.shape(indices)[0])),
                                       dense_shape=(num_samples, self.n_markers))
            mask = tf.sparse.add(mask, b)

        sparsified_data = tf.math.add(tf.math.multiply(x, mask), -1 * missing_value * (mask - 1))
        input_data_train = tf.stack([sparsified_data, mask], axis=-1)

        only_recomb = True

        if only_recomb and self.training and self.recombination_rate !=0 :
            input_data_train = input_data_train[self.batch_size:, :, :]
            x =  x[self.batch_size:, : ]
            tf.print("Observe, only passing recombined samples, no original samples. ")
        else:
            pass

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
        df = pd.read_parquet(self.filebase + ".parquet", columns=["0"])
        self.n_markers = len(df.to_numpy())

    def prep_normalizer(self, ds):
        # TODO- Fix so that the normalizer is calibrated with more samples than it is trained with (in each batch). In this case, if very few samples are used, there may be some issues if all have missing data at the same marker position
        temp = np.zeros((self.n_markers, 3))
        # temp = tf.zeros([self.n_markers,3])
        print(
            "Calibrating data scaler for normalization, remember to fix this. In prep_normalizer. Note in the function def.")
        if True or self.normalization_mode == "standard" or self.normalization_mode == "smartPCAstyle" or self.impute_missing == True:
            self.scaler = StandardScaler()

            for i, j, _ in ds:

                # Extract correct batch size, don't want to use recombined samples.
                i = i[:self.n_markers, :self.batch_size]

                # a = tf.zeros(shape = tf.shape(i))

                I = tf.convert_to_tensor(i)
                a = tf.sparse.SparseTensor(
                    indices=tf.cast(tf.where(I == 9), tf.int64),
                    values=(tf.repeat(np.nan, tf.shape(tf.where(I == 9))[0])),
                    dense_shape=tf.cast(tf.shape(tf.convert_to_tensor(i)), tf.int64)
                )

                if self.normalization_options["flip"]:
                    i = -(i - 2)  # This sends 1 to 1, 0 to 2, and 2 to 0
                self.scaler.partial_fit(tf.transpose(tf.sparse.add(a, i)))

                for k in range(3):
                    temp[:, k] += tf.reduce_sum(tf.cast(i == k, tf.int32), axis=1)

            self.most_common_genos = tf.cast(tf.transpose(np.argmax(temp, axis=1)), tf.float32)
            self.baseline_concordance = tf.reduce_mean(tf.reduce_max(temp, axis=1) / self.n_train_samples)

            # Compute baseline concordances for k-mers
            p_cor = tf.reduce_max(temp, axis=1) / self.n_train_samples
            k_vec = [1, 2, 3, 4, 5]
            for k in k_vec:
                p_cor2 = tf.concat([tf.reshape(p_cor, [self.n_markers, 1]), [[-1]]], axis=0)
                if k == k_vec[0]:
                    baseline_concordances_k_mer = [
                        tf.reduce_mean(tf.reduce_prod(tf.stack([p_cor2[i:-k + i, 0] for i in range(k)]), axis=0))]
                else:
                    baseline_concordances_k_mer = tf.concat([baseline_concordances_k_mer, [
                        tf.reduce_mean(tf.reduce_prod(tf.stack([p_cor2[i:-k + i, 0] for i in range(k)]), axis=0))]],
                                                            axis=0)

            self.baseline_concordances_k_mer = baseline_concordances_k_mer
            """

            This below is the how often we are correct if we choose most common variant, for all snp values
            tf.reduce_max(temp, axis=1) / self.n_train_samples

            """

            tf.print("Baseline concordance:", self.baseline_concordance)
            for i in range(5):
                tf.print("Baseline concordance {}-mer:".format(k_vec[i]), self.baseline_concordances_k_mer[i])

            if self.normalization_mode == "smartPCAstyle":
                p = (1 + self.scaler.mean_ * self.scaler.n_samples_seen_) / (2 + 2 * self.scaler.n_samples_seen_)

                self.scaler.var_ = p * (1 - p)

    def create_dataset(self, pref_chunk_size, mode, shuffle=True):
        if mode == "training":
            self.training = True
        elif mode == "validation":
            self.training = False
        else:
            print("pass either training or validation")
            return -1

        ds = tf.data.Dataset.from_generator(self.generator,
                                            output_signature=(
                                            tf.TensorSpec(shape=(self.n_markers, None), dtype=tf.float32),
                                            tf.TensorSpec(shape=(None, 2), dtype=tf.string),
                                            tf.TensorSpec(shape=(2,), dtype=tf.bool)),
                                            args=[pref_chunk_size, self.training, shuffle])
        if mode == "training":
            self.scaler = StandardScaler()
            self.scaler.mean_ = 0
            self.scaler.var = 1
            self.prep_normalizer(
                ds)  # This generates the needed scalers for normalization´using standard and smartPCAstyle
            self.most_common_genos = tf.ones([self.n_markers, ])

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)

        ds = ds.prefetch(tf.data.AUTOTUNE)

        if self.recombination and self.training:
            ds = ds.map(self.recombination_mapping, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.map(self.normalize_mapping, num_parallel_calls=tf.data.AUTOTUNE)
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

    elif os.path.isfile(filebase + ".bed"):
        print("Creating .parquet file from PLINK for faster data extraction. This may take a while")
        # Step 1, partition the dataset indices according to the maximum readable amount
        # The shape of the .bed files are given as (n_samples x n_markers), read as dtype = float32 = 8 Bytes

        if not os.path.isdir("Data_temp"):
            os.mkdir("Data_temp")

        snpreader = Bed(filebase + ".bed", count_A1=True)

        try:

            snpreader = Bed(filebase + ".bed", count_A1=False)
            # if this next line throws an error, the bim/fam file may use something like spaces instead of tabs

            (n_samples, n_markers) = snpreader.shape


        except:
            # Fix and create new bim file
            a = pd.read_csv(filebase + ".bim", header=None)
            b = a.to_numpy()
            c = [b[i][0].split() for i in range(len(b))]
            d = np.array(c)
            np.savetxt("data_temp/" + filebase + ".bim", d, delimiter="\t", fmt="%s")

            # Fix and create new fam file
            a = pd.read_csv(filebase + ".fam", header=None)
            b = a.to_numpy()
            c = [b[i][0].split() for i in range(len(b))]
            d = np.array(c)
            np.savetxt("data_temp/" + filebase + ".fam", d, delimiter="\t", fmt="%s")

            # Create a copy of the bed file, with the new name
            # I create new files as to not
            shutil.copyfile(filebase + ".bed", "data_temp/" + filebase + ".bed")

            snpreader = Bed("data_temp/" + filebase + ".bed", count_A1=False)

            (n_samples, n_markers) = snpreader.shape

        n_samples_in_chunk = np.floor(max_mem_size / 8.0 / n_samples)

        n_chunks = np.ceil(n_markers / n_samples_in_chunk)

        # Step 2, While looping over the chunks of indices, read the data into numpy arrays

        for i in range(int(n_chunks)):
            a = snpreader[:, int(n_samples_in_chunk * i):int(n_samples_in_chunk * (i + 1))].read().val

            # Step 3, Save each ndarray into .parquet file
            df = pd.DataFrame(a.T)
            table = pa.Table.from_pandas(df)
            it_number = 8 - len(str(i))
            if n_chunks > 1:
                pq.write_table(table, "Data_temp/temp" + it_number * "0" + str(i) + ".parquet")
            else:
                pq.write_table(table, filebase + ".parquet")

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

    elif os.path.isfile(filebase + ".eigenstratgeno"):
        print("Creating .parquet file from EIGENSTRAGENO for faster data extraction. This may take a while")

        n_samples = len(get_ind_pop_list(filebase))

        genotypes = np.genfromtxt(filebase + ".eigenstratgeno", delimiter=np.repeat(1, n_samples))

        df = pd.DataFrame(genotypes)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filebase + ".parquet")


