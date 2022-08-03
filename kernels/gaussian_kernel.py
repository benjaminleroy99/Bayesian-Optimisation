import numpy as np

from kernels.abstract_kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self,
                 log_amplitude: float,
                 log_length_scale: float,
                 log_noise_scale: float,
                 ):
        super(GaussianKernel, self).__init__(log_amplitude,
                                             log_length_scale,
                                             log_noise_scale,
                                             )

    def get_covariance_matrix(self,
                              X: np.ndarray,
                              Y: np.ndarray,
                              ) -> np.ndarray:
        """
        :param X: numpy array of size n_1 x m for which each row (x_i) is a data point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """

        n_1=X.shape(0)
        n_2=Y.shape(0)
        matrix=np.zeros((n_1,n_2))

        ln_sigma_f=GaussianKernel.log_amplitude
        ln_l=GaussianKernel.log_length_scale

        for i in range(n_1):
            for j in range(n_2):
                matrix[i,j]=np.exp(ln_sigma_f)**2*np.exp(-np.linalg.norm(X[i]-Y[j])**2/(2*np.exp(ln_l)**2))

        return matrix

    def __call__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 ) -> np.ndarray:
        return self.get_covariance_matrix(X, Y)

