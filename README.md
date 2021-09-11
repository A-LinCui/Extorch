![](https://img.shields.io/badge/version-1.0.0-yellow)
## Extorch: An useful extension library of PyTorch.

[**Extorch**](https://github.com/A-LinCui/Extorch) is an extension library of [**PyTorch**](https://github.com/pytorch/pytorch) that lets you easily build deep learning systems with PyTorch. 

📖 Documentation
-----------------

- **Tutorial**: If you are looking for a tutorial, check out examples under ``./example``.
- **Documentation**: The API documentation can be found on [ReadTheDocs](https://extorch.readthedocs.io/en/latest/).


🚀 Quickstart
--------------
As the API of this project may alter frequently in its early days, we recommand use the newest version at the [GitHub](https://github.com/A-LinCui/Extorch) following two steps below.

Step 1: ``git clone https://github.com/A-LinCui/Extorch``

Step 2: ``bash install.sh`` 

If you'd like to use the previous stable version.
Simply run ``pip install extorch`` in the command line.

🎉 Example
-----------
```
    import extorch.vision.dataset as dataset
    import torch.utils.data as data

    BATCH_SIZE = 128
    NUM_WORKERS = 2

    data_dir = "~/data" # Path to load the dataset
    datasets = dataset.CIFAR10(data_dir) # Construct the CIFAR10 dataset with standard transforms.

    trainloader = data.DataLoader(dataset = datasets.splits()["train"], \
            batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)
    testloader = data.DataLoader(dataset = datasets.splits()["test"], \
            batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = False)
```
More examples can be found in the ``./example`` folder.

👍 Contributions
-----------------
We welcome contributions of all kind.