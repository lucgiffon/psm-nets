import logging

from keras.preprocessing.image import ImageDataGenerator

from palmnet.core.palminizer import Palminizer
from palmnet.core.sparse_factorizer import SparseFactorizer
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.data_structures import SparseFactors
from qkmeans.palm.palm_fast import hierarchical_palm4msa, palm4msa

import numpy as np
from skimage.util import view_as_windows
from multiprocessing import Queue
from multiprocessing import Process

from skluc.utils import logger


class ActivationPalminizer(Palminizer):
    def __init__(self, train_data, batch_size, nb_epochs, queue_maxisize=2, val_data=None, sample_rate_conv=None, seed=None, *args, **kwargs):
        self.train_data = train_data
        self.val_data = val_data
        self.model_preprocessing = None
        self.sample_rate_conv = sample_rate_conv
        self.batch_size = batch_size
        self.seed = seed
        self.nb_epochs = nb_epochs
        self.queue_maxisize = queue_maxisize

        super().__init__(*args, **kwargs)

    def batch_preprocessing(self, data, block_size):
        if self.model_preprocessing is not None:
            processed_data = self.model_preprocessing.predict(data)
        else:
            processed_data = data

        if block_size is not None:
            if self.sample_rate_conv is None:
                nb_elm_by_patch = np.prod(block_size)
                sample_rate = 1 / nb_elm_by_patch
            else:
                sample_rate = self.sample_rate_conv
            window_shape = (1, *block_size, processed_data.shape[-1])
            processed_data = view_as_windows(processed_data, window_shape, step=1)
            processed_data = np.reshape(processed_data, (-1, np.prod(window_shape)))
            nb_elm_to_sample = int(processed_data.shape[0] * sample_rate)
            idx_to_keep = np.random.permutation(processed_data.shape[0])[:nb_elm_to_sample]
            processed_data = processed_data[idx_to_keep]
        return processed_data

    def factorize_one_batch_of_activations(self, processed_data, weight_matrix, lst_init_factors, init_lambda):
        matrix = processed_data @ weight_matrix

        logging.debug("Applying palm function to matrix with shape {}".format(matrix.shape))
        transposed = False

        # todo can be done outside the loop and function call
        if matrix.shape[0] > matrix.shape[1]:
            # we want the bigger dimension to be on right due to the residual computation that should remain big
            matrix = matrix.T
            transposed = True
            weight_matrix = weight_matrix.T

        left_dim, right_dim = weight_matrix.shape
        A = min(left_dim, right_dim)
        B = max(left_dim, right_dim)
        assert A == left_dim and B == right_dim, "Dimensionality problem: left dim should be higher than right dim before palm"

        if self.nb_factor is None:
            nb_factors = int(np.log2(B))
        else:
            nb_factors = self.nb_factor

        if lst_init_factors is None:
            lst_factors = [np.eye(A) for _ in range(nb_factors)]
            lst_factors[-1] = np.zeros((A, B))
        else:
            lst_factors = lst_init_factors

        if transposed:
            if lst_init_factors is not None:
                lst_factors = SparseFactors(lst_factors).transpose().get_list_of_factors()
            lst_factors = lst_factors + [processed_data.T]
        else:
            lst_factors = [processed_data] + lst_factors

        _lambda = init_lambda

        # todo it can be done before the function call, outside the loop
        lst_proj_op_by_fac_step, lst_proj_op_by_fac_step_desc = build_constraint_set_smart(left_dim=left_dim,
                                                                                           right_dim=right_dim,
                                                                                           nb_factors=nb_factors + 1,
                                                                                           sparsity_factor=self.sparsity_fac,
                                                                                           residual_on_right=True,
                                                                                           fast_unstable_proj=self.fast_unstable_proj,
                                                                                           constant_first=True,
                                                                                           hierarchical=self.hierarchical)

        if self.hierarchical:
            raise NotImplementedError
        elif transposed:
            lst_proj_op_by_fac_step = lst_proj_op_by_fac_step[1:] + [lst_proj_op_by_fac_step[0]]
            lst_proj_op_by_fac_step_desc = lst_proj_op_by_fac_step_desc[1:] + [lst_proj_op_by_fac_step_desc[0]]

        if self.hierarchical:
            final_lambda, final_factors, final_X, _, _ = hierarchical_palm4msa(
                arr_X_target=matrix,
                lst_S_init=lst_factors,
                lst_dct_projection_function=lst_proj_op_by_fac_step,
                f_lambda_init=_lambda,
                nb_iter=self.nb_iter,
                update_right_to_left=True,
                residual_on_right=True,
                delta_objective_error_threshold_palm=self.delta_threshold_palm,)
        else:
            final_lambda, final_factors, final_X, _, _ = \
                palm4msa(arr_X_target=matrix,
                         lst_S_init=lst_factors,
                         nb_factors=len(lst_factors),
                         lst_projection_functions=lst_proj_op_by_fac_step,
                         f_lambda_init=1.,
                         nb_iter=self.nb_iter,
                         update_right_to_left=True,
                         delta_objective_error_threshold=self.delta_threshold_palm,
                         track_objective=False)
            final_X *= final_lambda  # added later because palm4msa actually doesn't return the final_X multiplied by lambda contrary to hierarchical

        if transposed:
            final_lambda, final_factors, final_X = final_lambda, final_factors.transpose(), final_X.T
            weight_matrix = weight_matrix.T

        lst_factors_final = final_factors.get_list_of_factors(copy=True)[1:]
        final_factors = SparseFactors(lst_factors_final)
        final_X = final_lambda * final_factors.compute_product()

        print(np.linalg.norm(processed_data @ final_X - processed_data @ weight_matrix) / np.linalg.norm(processed_data @ weight_matrix))

        return final_lambda, final_factors, final_X

    @staticmethod
    def factorize_one_batch_of_activations_worker(fct_factorize_one_batch_of_activations, queue_palm, queue_result, queue_exception):
        try:
            logger.debug("Start factorize_one_batch_of_activations worker process.")
            processed_batch = queue_palm.get(block=True)
            current_factors = None
            current_lambda = 1.
            current_X = None
            while processed_batch is not None:
                logger.debug("Found one batch to process in factorize_one_batch_of_activations.")
                current_lambda, current_factors, current_X = fct_factorize_one_batch_of_activations(processed_batch, current_factors, current_lambda)
                current_factors = current_factors.get_list_of_factors()
                processed_batch = queue_palm.get(block=True)

            logger.info("Found None in factorize_one_batch_of_activations worker process. End process and deliver result.")
            queue_result.put((current_lambda, current_factors, current_X), block=True)
        except Exception as e:
            import sys
            queue_exception.put((e, str(sys.exc_info())))


    def apply_factorization(self, weight_matrix, block_size=None):
        """
        Apply Hierarchical-PALM4MSA algorithm to the input matrix and return the reconstructed approximation from
        the sparse factorisation.

        :param matrix: The matrix to apply PALM to.
        :param sparsity_fac: The sparsity factor for PALM.
        :return:
        """
        # multiprocessing will be done later
        logging.info("Applying factorization function to matrix with shape {}".format(weight_matrix.shape))
        data_gen = ImageDataGenerator(horizontal_flip=True,
                                      width_shift_range=0.125,
                                      height_shift_range=0.125,
                                      fill_mode='constant',
                                      cval=0.) # create bacthes from data


        nb_iter_by_epoch = self.train_data.shape[0] // self.batch_size

        if self.val_data is not None:
            processed_val_data = self.batch_preprocessing(self.val_data, block_size)
            val_activations = processed_val_data @ weight_matrix
            val_activations_norm = np.linalg.norm(val_activations)
            lst_diff = []
            print("After join")

        queue_palm = Queue(maxsize=self.queue_maxisize)
        queue_result = Queue(maxsize=self.queue_maxisize)
        queue_exception = Queue(maxsize=1)

        lambda_factorize_one_batch_of_activations_worker = lambda processed_batch, current_factors, current_lambda: self.factorize_one_batch_of_activations(processed_batch, weight_matrix, current_factors, current_lambda)
        process_palm = Process(target=self.factorize_one_batch_of_activations_worker, args=(lambda_factorize_one_batch_of_activations_worker, queue_palm, queue_result, queue_exception))
        process_palm.start()

        try:
            for i in range(self.nb_epochs):
                logger.debug(f"Epoch {i}")
                for i_iter, batch in enumerate(data_gen.flow(self.train_data, batch_size=self.batch_size, seed=self.seed)):
                    logger.debug(f"Process batch {i_iter}/{nb_iter_by_epoch}")
                    queue_palm.put(self.batch_preprocessing(batch, block_size))  # the palm process will handle automatically the batch from there
                    if i_iter+1 > nb_iter_by_epoch:
                        break
                    if not queue_exception.empty():
                        queue_palm.cancel_join_thread()
                        queue_result.cancel_join_thread()
                        base_exception, traceback_str = queue_exception.get(block=False)
                        str_exception = f"Got exception {str(base_exception)}\n Traceback was {traceback_str}"
                        raise Exception(str_exception)

            queue_palm.put(None, block=True)

            current_lambda, current_factors, current_X = queue_result.get(block=True)

        finally:
            process_palm.join()

        if self.val_data is not None:
            val_activations_pred = processed_val_data @ current_X
            diff = np.linalg.norm(val_activations - val_activations_pred) / val_activations_norm
            lst_diff.append(diff)

        current_factors = SparseFactors(current_factors)
        return current_lambda, current_factors, current_X


    def factorize_conv2D_weights(self, layer_weights):
        filter_height, filter_width, in_chan, out_chan = layer_weights.shape
        filter_matrix = layer_weights.reshape(filter_height * filter_width * in_chan, out_chan)
        _lambda, op_sparse_factors, reconstructed_filter_matrix = self.apply_factorization(filter_matrix, block_size=(filter_height, filter_width))
        new_layer_weights = reconstructed_filter_matrix.reshape(filter_height, filter_width, in_chan, out_chan)
        return _lambda, op_sparse_factors, new_layer_weights

    def set_preprocessing_model(self, model):
        if model is not None:
            model.compile("adam", loss="mse")
            self.model_preprocessing = model
        else:
            self.model_preprocessing = None
