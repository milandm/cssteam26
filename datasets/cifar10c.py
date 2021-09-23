from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CIFAR10C(VisionDataset):
    """CIFAR10C Dataset

    Reference:
    https://github.com/hendrycks/robustness
    https://arxiv.org/abs/1903.12261

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = '.'
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
    filename = "CIFAR-10-C.tar"
    # tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    files_list = [
        "brightness.npy",
        "gaussian_noise.npy",
        "saturate.npy",
        "contrast.npy",
        "glass_blur.npy",
        "shot_noise.npy",
        "defocus_blur.npy",
        "impulse_noise.npy",
        "snow.npy",
        "elastic_transform.npy",
        "jpeg_compression.npy",
        "spatter.npy",
        "fog.npy",
        "speckle_noise.npy",
        "frost.npy",
        "motion_blur.npy",
        "zoom_blur.npy",
        "gaussian_blur.npy",
        "pixelate.npy",
        "labels.npy"
    ]

    # meta = {
    #     'filename': 'batches.meta',
    #     'key': 'label_names',
    #     'md5': '5ff9c542aee3614f3951f8cda6e48888',
    # }

    def __init__(
            self,
            root: str,
            corruptions: list,
            severities: list,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10C, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        # if download:
        #     self.download()
        #
        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        self.corruptions = corruptions
        self.severities = severities
        self.data: Any = []
        self.targets = []

        self.labels = np.load(os.path.join(self.root, self.base_folder, 'labels.npy'))
        # now load the picked numpy arrays
        for idx, file_name in enumerate(self.corruptions):
            file_path = os.path.join(self.root, self.base_folder, file_name)
            corruption = np.load(file_path)
            severity = self.severities[idx]
            self.data.append(corruption[(severity - 1) * 10000:(severity * 10000)])
            self.targets.append(self.labels[(severity - 1) * 10000:(severity * 10000)])

        self.data = np.vstack(self.data)    #.reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.vstack(self.targets)[0]
        self.targets = self.targets.tolist()
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # self._load_meta()

    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta['filename'])
    #     with open(path, 'rb') as infile:
    #         data = pickle.load(infile, encoding='latin1')
    #         self.classes = data[self.meta['key']]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    # def _check_integrity(self) -> bool:
    #     root = self.root
    #     for fentry in (self.train_list + self.test_list):
    #         filename, md5 = fentry[0], fentry[1]
    #         fpath = os.path.join(root, self.base_folder, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True
    #
    # def download(self) -> None:
    #     # if self._check_integrity():
    #     #     print('Files already downloaded and verified')
    #     #     return
    #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
    #
    # def extra_repr(self) -> str:
    #     return "Split: {}".format("Train" if self.train is True else "Test")
