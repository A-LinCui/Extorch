from torchvision import transforms as T


class DetCompose(T.Compose):
    r"""
    Transform compose for detection.
    """
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
