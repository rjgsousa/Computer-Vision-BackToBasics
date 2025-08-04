
import numpy as np

def plot_gauss(x, y):
    import matplotlib.pyplot as plt

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10')
    plt.title("Multivariate Gaussian Data per Class")
    plt.show()


def gen_gauss(mean, sigma, samples_per_class) -> (np.ndarray, np.ndarray):
    _d1 = []  # data
    _y1 = []  # label

    for ix, samples in enumerate(samples_per_class):
        _d = np.random.multivariate_normal(mean, sigma, samples)
        _y = ix * np.ones((samples,1))

        _d1.append(_d)
        _y1.append(_y)

    _d1 = np.vstack(_d1)
    _y1 = np.vstack(_y1)
    _d1 = _d1 + 0.25 * np.random.rand(sum(samples_per_class),2)

    return _d1, _y1

def data_generation(samples_per_class: np.array) -> ((np.array, np.array),(np.array, np.array)):
    """

    Args:
        samples_per_class: number of samples per class

    Returns:
        a tuple of two np.array datasets.
        Each entry of the tuple consists of 2D Gauss data and class
    """

    sigma = np.array(((0.3,0),(0, 0.5)), dtype=float)
    mean = np.array((0,1), dtype=float)

    _dataset_1 = gen_gauss(mean, sigma, samples_per_class)
    _dataset_2 = gen_gauss(mean, sigma, np.int8(samples_per_class * .4))

    return _dataset_1, _dataset_2

if __name__ == '__main__':
    # generation of two datasets
    dataset_1, dataset_2 = data_generation(np.array([70, 50]))
    plot_gauss(dataset_1[0], dataset_1[1])
    plot_gauss(dataset_2[0], dataset_2[1])



