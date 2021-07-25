# -*- coding: utf-8 -*-

import os


class CapsNetParam(object):
    """A Container for the hyperparamters of Efficient-CapsNet.

    Attributes:
    """

    __slots__ = [
        "input_width",
        "input_height",
        "input_channel",
        "conv1_filter",
        "conv1_kernel",
        "conv1_stride",
        "conv2_filter",
        "conv2_kernel",
        "conv2_stride",
        "conv3_filter",
        "conv3_kernel",
        "conv3_stride",
        "conv4_filter",
        "conv4_kernel",
        "conv4_stride",
        "dconv_filter",
        "dconv_kernel",
        "dconv_stride",
        "num_primary_caps",
        "dim_primary_caps",
        "num_digit_caps",
        "dim_digit_caps",
    ]

    def __init__(self,
                 input_width: int = 28,
                 input_height: int = 28,
                 input_channel: int = 1,
                 conv1_filter: int = 32,
                 conv1_kernel: int = 5,
                 conv1_stride: int = 1,
                 conv2_filter: int = 64,
                 conv2_kernel: int = 3,
                 conv2_stride: int = 1,
                 conv3_filter: int = 64,
                 conv3_kernel: int = 3,
                 conv3_stride: int = 1,
                 conv4_filter: int = 128,
                 conv4_kernel: int = 3,
                 conv4_stride: int = 2,
                 dconv_kernel: int = 9,
                 dconv_stride: int = 1,
                 num_primary_caps: int = 16,
                 dim_primary_caps: int = 8,
                 num_digit_caps: int = 10,
                 dim_digit_caps: int = 16,
                 *args,
                 **kwargs) -> None:

        # Input Specification
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel

        # FeatureMap Layer
        self.conv1_filter = conv1_filter
        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride
        self.conv2_filter = conv2_filter
        self.conv2_kernel = conv2_kernel
        self.conv2_stride = conv2_stride
        self.conv3_filter = conv3_filter
        self.conv3_kernel = conv3_kernel
        self.conv3_stride = conv3_stride
        self.conv4_filter = conv4_filter
        self.conv4_kernel = conv4_kernel
        self.conv4_stride = conv4_stride

        # PrimaryCap Layer
        self.dconv_filter = num_primary_caps * dim_primary_caps
        self.dconv_kernel = dconv_kernel
        self.dconv_stride = dconv_stride
        self.num_primary_caps = num_primary_caps
        self.dim_primary_caps = dim_primary_caps

        # DigitCap Layer
        self.num_digit_caps = num_digit_caps
        self.dim_digit_caps = dim_digit_caps

    def get_config(self) -> dict:
        return {
            "input_width": self.input_width,
            "input_height": self.input_height,
            "input_channel": self.input_channel,
            "conv1_filter": self.conv1_filter,
            "conv1_kernel": self.conv1_kernel,
            "conv1_stride": self.conv1_stride,
            "conv2_filter": self.conv2_filter,
            "conv2_kernel": self.conv2_kernel,
            "conv2_stride": self.conv2_stride,
            "conv3_filter": self.conv3_filter,
            "conv3_kernel": self.conv3_kernel,
            "conv3_stride": self.conv3_stride,
            "conv4_filter": self.conv4_filter,
            "conv4_kernel": self.conv4_kernel,
            "conv4_stride": self.conv4_stride,
            "dconv_filter": self.dconv_filter,
            "dconv_kernel": self.dconv_kernel,
            "dconv_stride": self.dconv_stride,
            "num_primary_caps": self.num_primary_caps,
            "dim_primary_caps": self.dim_primary_caps,
            "num_digit_caps": self.num_digit_caps,
            "dim_digit_caps": self.dim_digit_caps
        }

    def save_config(self, path: str) -> None:
        """Saves configuration.
        
        Collects attributes as pair of name and value and saves them to a UTF-8
        encoded file.
        Args:
            path (str): A filepath to write configuration. If any file already 
                exists, its contents will be overwritten.
        
        Raises:
            TypeError: If `path` is not string.
            ValueError: If `path` is empty.
        """
        if not isinstance(path, str):
            raise TypeError()
        elif len(path) == 0:
            raise ValueError()
        else:
            with open(path, 'w', encoding='utf8') as f:
                for k, v in self.get_config().items():
                    f.writelines(f"{k}={v}\n")


def load_config(path: str) -> CapsNetParam:
    """Loads configuration.
        
    Reads file with the given path and makes `CapsNetParam` instance by parsing
    the contents of the file.

    Args:
        path (str): A filepath to read configuration.
    
    Returns:
        A `CapsNetParam` instance.

    Raises:
        TypeError: If `path` is not string.
        ValueError: If `path` is empty.
        FileNotFoundError: If file of `path` not exists.
    """
    if not isinstance(path, str):
        raise TypeError()
    elif len(path) == 0:
        raise ValueError()
    elif not os.path.isfile(path):
        raise FileNotFoundError()

    with open(path, 'r', encoding="utf8") as f:
        config = []
        for l in f.readlines():
            k, v = l.strip().split('=')
            config.append((k, int(v)))
        return CapsNetParam(**dict(config))


def make_param(image_width: int = 28,
               image_height: int = 28,
               image_channel: int = 1,
               conv1_filter: int = 32,
               conv1_kernel: int = 5,
               conv1_stride: int = 1,
               conv2_filter: int = 64,
               conv2_kernel: int = 3,
               conv2_stride: int = 1,
               conv3_filter: int = 64,
               conv3_kernel: int = 3,
               conv3_stride: int = 1,
               conv4_filter: int = 128,
               conv4_kernel: int = 3,
               conv4_stride: int = 2,
               dconv_kernel: int = 9,
               dconv_stride: int = 1,
               num_primary_caps: int = 16,
               dim_primary_caps: int = 8,
               num_digit_caps: int = 10,
               dim_digit_caps: int = 16) -> CapsNetParam:
    return CapsNetParam(
        image_width,
        image_height,
        image_channel,
        conv1_filter,
        conv1_kernel,
        conv1_stride,
        conv2_filter,
        conv2_kernel,
        conv2_stride,
        conv3_filter,
        conv3_kernel,
        conv3_stride,
        conv4_filter,
        conv4_kernel,
        conv4_stride,
        dconv_kernel,
        dconv_stride,
        num_primary_caps,
        dim_primary_caps,
        num_digit_caps,
        dim_digit_caps,
    )
