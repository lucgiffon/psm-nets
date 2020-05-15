import numpy as np
import scipy.stats
from keras import backend as K

from skluc.utils.datautils import build_hadamard


def G_variable(shape, trainable=False):
    """
    Return a Gaussian Random matrix converted into Tensorflow Variable.

    :param shape: The shape of the matrix (number of fastfood stacks (v), dimension of the input space (d))
    :type shape: int or tuple of int (tuple size = 2)
    :return: K.variable object containing the matrix, The norm2 of each line (np.array of float)
    """
    assert type(shape) == int or (type(shape) == tuple and len(shape) == 2)
    G = np.random.normal(size=shape).astype(np.float32)
    G_norms = np.linalg.norm(G, ord=2, axis=1)
    return G, G_norms
    # if trainable:
    #     return K.variable(G, name="G"), G_norms
    # else:
    #     return K.constant(G, name="G"), G_norms


def B_variable(shape, trainable=False):
    """
    Return a random matrix of -1 and 1 picked uniformly and converted into Tensorflow Variable.

    :param shape: The shape of the matrix (number of fastfood stacks (v), dimension of the input space (d))
    :type shape: int or tuple of int (tuple size = 2)
    :return: K.variable object containing the matrix
    """
    assert type(shape) == int or (type(shape) == tuple and len(shape) == 2)
    B = np.random.choice([-1, 1], size=shape, replace=True).astype(np.float32)
    if trainable:
        return K.variable(B, name="B")
    else:
        return K.constant(B, name="B")


def P_variable(d, nbr_stack):
    """
    Return a permutation matrix converted into Tensorflow Variable.

    :param d: The width of the matrix (dimension of the input space)
    :type d: int
    :param nbr_stack: The height of the matrix (nbr_stack x d is the dimension of the output space)
    :type nbr_stack: int
    :return: K.variable object containing the matrix
    """
    idx = np.hstack([(i * d) + np.random.permutation(d) for i in range(nbr_stack)])
    P = np.eye(N=nbr_stack * d)[idx].astype(np.float32)
    return K.constant(P, name="P")


def H_variable(d):
    """
    Return an Hadamard matrix converted into Tensorflow Variable.

    d must be a power of two.

    :param d: The size of the Hadamard matrix (dimension of the input space).
    :type d: int
    :return: K.variable object containing the diagonal and not trainable
    """
    H = build_hadamard(d).astype(np.float32)
    return K.constant(H, name="H")


def S_variable(shape, G_norms, trainable=False):
    """
    Return a scaling matrix of random values picked from a chi distribution.

    The values are re-scaled using the norm of the associated Gaussian random matrix G. The associated Gaussian
    vectors are the ones generated by the `G_variable` function.

    :param shape: The shape of the matrix (number of fastfood stacks (v), dimension of the input space (d))
    :type shape: int or tuple of int (tuple size = 2)
    :param G_norms: The norms of the associated Gaussian random matrices G.
    :type G_norms: np.array of floats
    :return: K.variable object containing the matrix.
    """
    S = np.multiply((1 / G_norms.reshape((-1, 1))), scipy.stats.chi.rvs(shape[1], size=shape).astype(np.float32))
    # if trainable:
    #     return K.variable(S, name="S")
    # else:
    #     return K.constant(S, name="S")
    return S