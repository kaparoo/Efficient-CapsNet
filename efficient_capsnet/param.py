# -*- coding: utf-8 -*-

class CapsNetParam(object):

    __slots__ = ("conv1_filter", "conv1_kernel", "conv1_stride",
                 "conv2_filter", "conv2_kernel", "conv2_stride",
                 "conv3_filter", "conv3_kernel", "conv3_stride",
                 "conv4_filter", "conv4_kernel", "conv4_stride",
                 "dconv_filter", "dconv_kernel", "dconv_stride",
                 "num_primary_caps", "dim_primary_caps", "num_digit_caps",
                 "dim_digit_caps")

    def __init__(self,
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

        # FeatureMap
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

        # PrimaryCaps
        self.dconv_filter = num_primary_caps * dim_primary_caps
        self.dconv_kernel = dconv_kernel
        self.dconv_stride = dconv_stride
        self.num_primary_caps = num_primary_caps
        self.dim_primary_caps = dim_primary_caps

        # DigitCaps
        self.num_digit_caps = num_digit_caps
        self.dim_digit_caps = dim_digit_caps

    def get_config(self) -> dict:
        return {
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
        with open(path, 'w', encoding='utf8') as f:
            for k, v in self.get_config().items():
                f.writelines(f"{k}={v}\n")

    @staticmethod
    def load_config(path: str) -> object:
        with open(path, 'r', encoding='utf8') as f:
            config = []
            for l in f.readlines():
                k, v = l.strip().split('=')
                config.append((k, int(v)))
            return CapsNetParam(**dict(config))


if __name__ == "__main__":
    old_param = CapsNetParam()
    old_param.save_config("test.txt")
    print(old_param.get_config())
    del old_param

    new_param = CapsNetParam.load_config("test.txt")
    print(new_param.get_config())
