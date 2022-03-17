from typing import List, Union

import numpy as np
from sklearn.manifold import TSNE
import torch
from torch import Tensor


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


def denormalize(
    image: Tensor, 
    mean: List[float], 
    std: List[float], 
    transpose: bool = False, 
    detach_numpy: bool = False) -> Union[Tensor, np.ndarray]:
    r"""
    De-normalize the tensor-like image.

    Args:
        image (Tensor): The image to be de-normalized with shape [B, C, H, W] or [C, H, W].
        mean (List[float]): Sequence of means for each channel while normalizing the origin image.
        std (List[float]): Sequence of standard deviations for each channel while normalizing the origin image.
        transpose (bool): Whether transpose the image to [B, H, W, C] or [H, W, C]. Default: `False`.
        detach_numpy (bool): If true, return `Tensor.detach().cpu().numpy()`.

    Returns:
        image (Union[Tensor, np.ndarray]): The de-normalized image.

    Examples:
        >>> image = torch.randn((5, 3, 32, 32)).cuda()  # Shape: [5, 3, 32, 32] (cuda)
        >>> mean = [0.5, 0.5, 0.5]
        >>> std = [1., 1., 1.]
        >>> de_image = denormalize(image, mean, std, True, True)  # Shape: [5, 32, 32, 3] (cpu)
    """
    device = image.device
    mean = torch.reshape(torch.tensor(mean), (3, 1, 1)).to(device)
    std = torch.reshape(torch.tensor(std), (3, 1, 1)).to(device)
    image = image * std + mean
    if transpose:
        image = image.permute(0, 2, 3, 1) if len(image.shape) == 4 else image.permute(1, 2, 0)
    if detach_numpy:
        image = image.detach().cpu().numpy()
    return image
