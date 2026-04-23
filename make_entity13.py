import torch
import torchvision.transforms as transforms
import numpy as np

from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from robustness.tools.breeds_helpers import (
    make_living17,
    make_entity13,
    make_entity30,
    make_nonliving26
)

DATA_DIR = "/home/arina/imagenet"
BATCH_SIZE = 64

IMAGENET_C = [
    "fog", "frost", "motion_blur", "brightness", "zoom_blur",
    "snow", "defocus_blur", "glass_blur", "gaussian_noise",
    "shot_noise", "impulse_noise", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression", "speckle_noise", "spatter",
    "gaussian_blur", "saturate"
]
SEVERITIES = [1, 2, 3, 4, 5]


def get_imagenet_breeds(batch_size, data_dir, name="living17"):

    hierarchy_dir = f"{data_dir}/imagenet_class_hierarchy"

    # --- Выбор датасета ---
    if name == "living17":
        ret = make_living17(hierarchy_dir, split="good")
    elif name == "entity13":
        ret = make_entity13(hierarchy_dir, split="good")
    elif name == "entity30":
        ret = make_entity30(hierarchy_dir, split="good")
    elif name == "nonliving26":
        ret = make_nonliving26(hierarchy_dir, split="good")
    else:
        raise ValueError(f"Unknown breeds name: {name}")

    # --- Label mappings ---
    source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0])
    target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4717, 0.4499, 0.3837],
                             [0.2600, 0.2516, 0.2575])
    ])

    # --- Train / Val split ---
    trainset = folder.ImageFolder(
        root=f"{data_dir}/imagenetv1/train/",
        transform=transform,
        label_mapping=source_label_mapping
    )
    targetset = folder.ImageFolder(
        root=f"{data_dir}/imagenetv1/train/",
        transform=transform,
        label_mapping=target_label_mapping
    )

    idx = np.arange(len(trainset))
    np.random.seed(42)
    np.random.shuffle(idx)

    train_idx = idx[:len(idx) - 10000]
    val_idx   = idx[len(idx) - 10000:]

    train_subset = torch.utils.data.Subset(trainset, train_idx)
    val_subset   = torch.utils.data.Subset(trainset, val_idx)

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    testsets    = []
    testloaders = []

    def add_loader(ds):
        testsets.append(ds)
        testloaders.append(
            torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=4
            )
        )

    # --- Clean sets ---
    # val из train (source / target)
    add_loader(val_subset)
    add_loader(targetset)

    # ImageNet val source
    add_loader(folder.ImageFolder(
        f"{data_dir}/imagenetv1/val/",
        transform=transform,
        label_mapping=source_label_mapping
    ))

    # ImageNet val target
    add_loader(folder.ImageFolder(
        f"{data_dir}/imagenetv1/val/",
        transform=transform,
        label_mapping=target_label_mapping
    ))

    # --- Corruptions SOURCE mapping ---
    print(f"\n  Загружаем corruptions (source)...")
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            path = f"{data_dir}/imagenet-c/{corruption}/{severity}"
            add_loader(folder.ImageFolder(
                root=path,
                transform=transform,
                label_mapping=source_label_mapping
            ))

    # --- Corruptions TARGET mapping ---
    print(f"  Загружаем corruptions (target)...")
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            path = f"{data_dir}/imagenet-c/{corruption}/{severity}"
            add_loader(folder.ImageFolder(
                root=path,
                transform=transform,
                label_mapping=target_label_mapping
            ))

    print(f"\n✅ BREEDS '{name}' готов!")
    print(f"   Train size       : {len(train_subset)}")
    print(f"   Val size         : {len(val_subset)}")
    print(f"   Source classes   : {len(ret[1][0])}")
    print(f"   Target classes   : {len(ret[1][1])}")
    print(f"   Total test sets  : {len(testsets)}")
    print(f"     - 4 clean sets")
    print(f"     - {len(IMAGENET_C) * len(SEVERITIES)} corruption sets (source)")
    print(f"     - {len(IMAGENET_C) * len(SEVERITIES)} corruption sets (target)")

    return trainset, trainloader, testsets, testloaders


if __name__ == "__main__":
    for breeds_name in ["living17", "entity13", "entity30", "nonliving26"]:
        print(f"\n{'='*50}")
        print(f"Создаём {breeds_name}...")
        print('='*50)

        trainset, trainloader, testsets, testloaders = get_imagenet_breeds(
            batch_size=BATCH_SIZE,
            data_dir=DATA_DIR,
            name=breeds_name
        )

        # Быстрая проверка
        images, labels = next(iter(trainloader))
        print(f"   Batch shape      : {images.shape}")
        print(f"   Labels sample    : {labels[:8].tolist()}")