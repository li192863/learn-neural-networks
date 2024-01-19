import torch
from torchvision.ops import nms
import numpy as np


class DecodeBox():
    """
    将网络输出解码为真实像素坐标
    """

    def __init__(self, anchors, num_classes, input_shape, device, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.device = device
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        """
        解码网络输出信息
        :param inputs: 网络输出信息 (nx255x13x13, nx255x26x26, nx255x52x52)，255为tx, ty, tw, th, conf, class1, ..., class_n
        :return: list，元素为(batch_size, 3 * input_width * input_height, 4 + 1 + num_classes)，最后维度为bx, by, bw, bh, conf, class1, ..., class_n
        """
        outputs = []
        for i, input in enumerate(inputs):
            # 输入的input一共有三个，他们的shape分别是(batch_size, 255, 13/26/52, 13/26/52)
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)
            # 输入为416x416时，stride_h = stride_w = 32、16、8
            stride_h = self.input_shape[0] / input_height  # 特征图缩放的倍数
            stride_w = self.input_shape[1] / input_width
            # 此时获得的scaled_anchors大小是相对于特征层的
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]
            # 输入的input一共有三个，他们的shape分别是(batch_size, 3, 13/26/52, 13/26/52, 85)
            prediction = input.view(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height,
                                    input_width).permute(0, 1, 3, 4, 2).contiguous()  # 将depth维度移至最后

            # 先验框的中心位置的调整参数
            x = torch.sigmoid(prediction[..., 0])  # (batch_size, 3, 13, 13)
            y = torch.sigmoid(prediction[..., 1])
            # 先验框的宽高调整参数
            w = prediction[..., 2]
            h = prediction[..., 3]
            conf = torch.sigmoid(prediction[..., 4])  # 获得置信度，是否有物体
            pred_cls = torch.sigmoid(prediction[..., 5:])  # 种类置信度

            # 生成网格，先验框中心，网格左上角
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).to(self.device)  # (batch_size, 3, 13, 13)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).to(self.device)

            # 按照网格格式生成先验框的宽高(batch_size, 3, 13, 13)
            anchor_w = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([0])).to(
                self.device)  # 沿着1轴选择index为0的元素
            anchor_h = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([1])).to(self.device)
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(
                w.shape)  # (batch_size, 3, 13, 13)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # 利用预测结果对先验框进行调整，首先调整先验框的中心，从先验框中心向右下角偏移，再调整先验框的宽高。
            pred_boxes = torch.FloatTensor(prediction[..., :4].shape).to(self.device)  # (batch_size, 13, 13, 4)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            # 将输出结果归一化成小数的形式
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).to(self.device)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,  # 归一化
                                conf.view(batch_size, -1, 1),
                                pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)  # (batch_size, 3 * input_width * input_height, 4 + 1 + num_classes)
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        """
        根据图片将边界框修正为像素值
        :param box_xy:
        :param box_wh:
        :param input_shape:
        :param image_shape:
        :param letterbox_image:
        :return: 4D tensor，最后维度(y1, x1, y2, x2)，单位为像素
        """
        # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # 这里求出来的offset是图像有效区域相对于图像左上角的偏移情况，new_shape指的是宽高缩放情况
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))  # height, width
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape  # height_scale, width_scale

            box_yx = (box_yx - offset) * scale
            box_hw *= scale
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)  # (y1, x1, y2, x2)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)  # height, width, height, width
        return boxes  # (y1, x1, y2, x2)

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image,
                            conf_threshold=0.5,
                            nms_threshold=0.4):
        """
        非极大值抑制
        :param prediction: 预测结果沿一维度堆叠的tensor (batch_size, num_anchors, 5 + num_classes)
        :param num_classes:
        :param input_shape:
        :param image_shape:
        :param letterbox_image:
        :param conf_threshold:
        :param nms_threshold:
        :return: list，元素为(anchors, 7)，共batch_size个元素
        """
        # 将预测结果的格式转换成左上角右下角的格式(batch_size, num_anchors, 85)
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # 左上角点x坐标x1
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y1
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # 右下角点x坐标x2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # 对种类预测部分取max，种类置信度class_conf为[num_anchors, 1]，种类class_pred为[num_anchors, 1]
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            # 利用置信度进行第一轮筛选
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_threshold).squeeze()
            # 根据置信度进行预测结果的筛选
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue

            # 检测结果detections为[num_anchors, 7]，7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            # 获得预测结果中包含的所有种类
            unique_labels = detections[:, -1].cpu().unique().to(self.device)

            for c in unique_labels:
                # 获得某一类得分筛选后全部的预测结果
                detections_class = detections[detections[:, -1] == c]

                # 使用官方自带的非极大抑制会速度更快一些！
                keep = nms(
                    detections_class[:, :4],  # boxes (Tensor[N, 4]))
                    detections_class[:, 4] * detections_class[:, 5],  # scores
                    nms_threshold  # iou_threshold
                )
                max_detections = detections_class[keep]

                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data

                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output  # list 元素(anchors, 7) 共batch_size个元素


if __name__ == '__main__':
    from util import get_anchors

    anchors = get_anchors('../data/yolo_anchors.txt')[0]

    input1 = torch.randn((1, 24, 13, 13)).to('cuda')
    input2 = torch.randn((1, 24, 26, 26)).to('cuda')
    input3 = torch.randn((1, 24, 52, 52)).to('cuda')
    inputs = input1, input2, input3

    decode_box = DecodeBox(anchors, 3, [416, 416], 'cuda')
    print(decode_box.decode_box(inputs)[0].shape)
    print(decode_box.decode_box(inputs)[1].shape)
    print(decode_box.decode_box(inputs)[2].shape)
