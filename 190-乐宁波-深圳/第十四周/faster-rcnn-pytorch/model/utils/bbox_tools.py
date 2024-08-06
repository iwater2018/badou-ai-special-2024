import numpy as np


def loc2bbox(src_bbox, loc):
    """
    从边界框偏移和尺度解码边界框。

    给定由 :meth:`bbox2loc` 计算出的边界框偏移和尺度，这个函数将表示解码为2D图像坐标中的坐标。

    给定尺度和偏移 :math:`t_y, t_x, t_h, t_w` 以及其中心为 :math:`(y, x) = p_y, p_x` 且大小为 :math:`p_h, p_w` 的边界框，
    解码后的边界框的中心 :math:`\\hat{g}_y`, :math:`\\hat{g}_x` 和大小 :math:`\\hat{g}_h`, :math:`\\hat{g}_w` 通过以下公式计算。

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    解码公式在诸如 R-CNN [#]_ 等工作中使用。

    输出与输入的类型相同。

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    准确的目标检测和语义分割的丰富特征层次。 CVPR 2014.

    参数：
        src_bbox (array): 边界框的坐标。
            其形状是 :math:`(R, 4)`。这些坐标是
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`。
        loc (array): 包含偏移和尺度的数组。
            :obj:`src_bbox` 和 :obj:`loc` 的形状应该相同。
            这包含值 :math:`t_y, t_x, t_h, t_w`。

    返回：
        array:
        解码后的边界框坐标。其形状是 :math:`(R, 4)`。
        第二轴包含四个值
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`。

    """
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.maximum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size=16, ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32)):
    """
    通过枚举宽高比和尺度生成基准锚点。

    生成根据给定的宽高比和尺度进行缩放和修改的锚点。
    在修改为给定宽高比时，缩放锚点的面积保持不变。

    此函数生成了 :obj:`R = len(ratios) * len(anchor_scales)` 个锚点。
    第 :obj:`i * len(anchor_scales) + j` 个锚点对应于由 :obj:`ratios[i]` 和 :obj:`anchor_scales[j]` 生成的锚点。

    例如，如果尺度是 :math:`8` 而宽高比是 :math:`0.25`，
    基准窗口的宽度和高度将被拉伸 :math:`8` 倍。
    为了将锚点修改为给定的宽高比，
    高度减半，宽度加倍。

    参数：
        base_size (number): 基准窗口的宽度和高度。
        ratios (list of floats): 这是锚点的宽度与高度的比率。
        anchor_scales (list of numbers): 这是锚点的面积。
            这些面积将是 :obj:`anchor_scales` 中一个元素的平方与基准窗口原始面积的乘积。

    返回：
        ~numpy.ndarray:
        一个形状为 :math:`(R, 4)` 的数组。
        每个元素是边界框的坐标集合。
        第二轴对应于边界框的 :math:`(y_{min}, x_{min}, y_{max}, x_{max})`。
    """
    # 中心点坐标
    py = base_size / 2.
    px = base_size / 2.
    print(py, px)

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            print(ratios, anchor_scales, h, w)

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


if __name__ == '__main__':
    a = generate_anchor_base()
    # print(a)
    bbox_a = np.array([[1, 1, 4, 4], [2, 3, 5, 4]])  # 两个边界框 A1 和 A2
    bbox_b = np.array([[2, 2, 3, 3]])
    r = bbox_iou(bbox_a, bbox_b)
    print(r)
