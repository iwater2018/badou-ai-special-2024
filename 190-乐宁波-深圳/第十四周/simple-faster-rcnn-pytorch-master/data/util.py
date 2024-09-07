import numpy as np
from PIL import Image
import random


def read_image(path, dtype=np.float32, color=True):
    """
    从文件中读取图像。

        该函数从给定的文件中读取图像。图像为CHW格式，其值的范围是 :math:`[0, 255]` 。如果 :obj:`color = True` ，通道的顺序是RGB。

        参数：
            path (str): 图像文件的路径。
            dtype: 数组的类型。默认值是 :obj:`~numpy.float32` 。
            color (bool): 此选项确定通道的数量。
                如果 :obj:`True` ，通道数量为三个。在这种情况下，通道的顺序是RGB。这是默认行为。
                如果 :obj:`False` ，此函数返回灰度图像。

        返回：
            ~numpy.ndarray: 一个图像。
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def resize_bbox(bbox, in_size, out_size):
    """根据图像的缩放调整边界框的大小。

    边界框应该是一个形状为 :math:`(R, 4)` 的二维张量，其中 :math:`R` 表示图像中的边界框数量。第二个轴表示边界框的属性。
    它们是 :math:`(y_{min}, x_{min}, y_{max}, x_{max})`，这四个属性分别是左上角和右下角顶点的坐标。

    参数:
        bbox (~numpy.ndarray): 一个形状为 :math:`(R, 4)` 的数组。:math:`R` 是边界框的数量。
        in_size (tuple): 一个长度为 2 的元组。图像缩放前的高度和宽度。
        out_size (tuple): 一个长度为 2 的元组。图像缩放后的高度和宽度。

    返回:
        ~numpy.ndarray:
        根据给定的图像尺寸重新调整的边界框。
    """

    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """
    翻转图像时
    相应地翻转边界框。

        预期边界框被打包进一个二维张量中，形状为 :math:`(R, 4)`，其中 :math:`R` 是图像中边界框的数量。第二轴表示边界框的属性。
        它们是 :math:`(y_{min}, x_{min}, y_{max}, x_{max})`，这四个属性分别是左上角和右下角顶点的坐标。

        参数：
            bbox (~numpy.ndarray): 一个形状为 :math:`(R, 4)` 的数组。
                :math:`R` 是边界框的数量。
            size (tuple): 长度为2的元组。表示图像调整大小时的高度和宽度。
            y_flip (bool): 根据图像的垂直翻转来翻转边界框。
            x_flip (bool): 根据图像的水平翻转来翻转边界框。

        返回：
            ~numpy.ndarray:
            根据给定的翻转方式翻转后的边界框。

    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def crop_bbox(
        bbox, y_slice=None, x_slice=None,
        allow_outside_center=True, return_param=False):
    """
    将边界框转换为适应图像裁剪区域。

    这个方法主要与图像裁剪一起使用。
    这个方法像 :func:`data.util.translate_bbox` 一样转换边界框的坐标。此外，
    这个函数会截断边界框以适应裁剪区域。
    如果一个边界框与裁剪区域没有重叠，这个边界框将被移除。

    预期边界框被打包进一个二维张量中，形状为 :math:`(R, 4)`，其中 :math:`R` 是图像中边界框的数量。第二轴表示边界框的属性。它们是 :math:`(y_{\text{min}}, x_{\text{min}}, y_{\text{max}}, x_{\text{max}})`，这四个属性分别是左上角和右下角顶点的坐标。

    参数：
        bbox (~numpy.ndarray): 要转换的边界框。形状为
            :math:`(R, 4)`。:math:`R` 是边界框的数量。
        y_slice (slice): Y轴的切片。
        x_slice (slice): X轴的切片。
        allow_outside_center (bool): 如果这个参数是 :obj:`False`，
            中心在裁剪区域外的边界框将被移除。默认值是 :obj:`True`。
        return_param (bool): 如果是 :obj:`True`，这个函数返回
            保留的边界框的索引。

    返回：
        ~numpy.ndarray 或 (~numpy.ndarray, dict):

        如果 :obj:`return_param = False`，返回数组 :obj:`bbox`。

        如果 :obj:`return_param = True`，
        返回一个元组，其元素是 :obj:`bbox, param`。
        :obj:`param` 是一个包含中间参数的字典，其内容下面列出了键、值类型和值的描述。

        * **index** (*numpy.ndarray*): 包含使用的边界框索引的数组。
    """

    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]) \
            .all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """Translate bounding boxes.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate
    :math:`(y, x) = (y_{offset}, x_{offset})`.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    Returns:
        ~numpy.ndarray:
        Bounding boxes translated according to the given offsets.

    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.

    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img
