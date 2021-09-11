import numpy as np
from sklearn.manifold import TSNE


def tsne_fit(feature: np.ndarray, n_components: int = 2, init: str = "pca", **kwargs):
    r"""
    Fit input features into an embedded space and return that transformed output.

    Args:
        feature (np.ndarray): The features to be embedded.
        n_components (int): Dimension of the embedded space. Default: 2.
        init (str): Initialization of embedding. Possible options are "random", "pca",
                    and a numpy array of shape (n_samples, n_components).
                    PCA initialization cannot be used with precomputed distances and is
                    usually more globally stable than random initialization.
                    Default: "pca".
        kwargs: Other configurations for TSNE model construction.
        
    Returns:
       node_pos (np.ndarray): The representation in the embedding space.

    Examples::
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> features = np.random.randn(50, 10)
        >>> labels = np.random.randint(0, 2, (50, 1))
        >>> node_pos = tsne_fit(features, 2, "pca")
        >>> plt.figure()
        >>> plt.scatter(node_pos[:, 0], node_pos[:, 1], c = labels)
        >>> plt.show()
    """
    model = TSNE(n_components = n_components, init = init, **kwargs)
    node_pos = model.fit_transform(feature)
    return node_pos
