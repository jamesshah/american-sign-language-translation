import albumentations
IMAGE_SIZE = 128


class Augments:
    """
    Contains Train and Test Augments
    """

    train_augments = albumentations.Compose([
        albumentations.Resize(
            IMAGE_SIZE, IMAGE_SIZE, always_apply=True),
        albumentations.Normalize(always_apply=True)
    ], p=1)

    valid_augments = albumentations.Compose([
        albumentations.Resize(
            IMAGE_SIZE, IMAGE_SIZE, always_apply=True),
        albumentations.Normalize(always_apply=True)
    ], p=1)
