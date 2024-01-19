import math

import numpy as np
import torch
from torch import nn


class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, device, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(YoloLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.device = device
        self.anchors_mask = anchors_mask

        self.ignore_threshold = 0.5

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        return -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)

    def forward(self, l, input, targets):
        # 输入的input一共有三个，他们的shape分别是(batch_size, 255, 13/26/52, 13/26/52)
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)
        # 输入为416x416时，stride_h = stride_w = 32、16、8
        stride_h = self.input_shape[0] / input_height
        stride_w = self.input_shape[1] / input_width
        # 此时获得的scaled_anchors大小是相对于特征层的
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        # 输入的input一共有三个，他们的shape分别是(batch_size, 3, 13/26/52, 13/26/52, 85)
        prediction = input.view(batch_size, len(self.anchors_mask[l]), self.bbox_attrs, input_height,
                                input_width).permute(0, 1, 3, 4, 2).contiguous()  # 将depth维度移至最后

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  # (batch_size, 3, 13, 13)
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])  # 获得置信度，是否有物体
        pred_cls = torch.sigmoid(prediction[..., 5:])  # 种类置信度

        # 获得网络应该有的预测结果
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, input_height, input_width)

        # 将预测结果进行解码，判断预测结果和真实值的重合程度
        noobj_mask = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, input_height, input_width, noobj_mask)
        y_true = y_true.to(self.device)
        noobj_mask = noobj_mask.to(self.device)
        box_loss_scale = box_loss_scale.to(self.device)
        # reshape_y_true[...,2:3]和reshape_y_true[...,3:4]表示真实框的宽高，二者均在0-1之间 真实框越大，比重越小，小框的比重更大。
        box_loss_scale = 2 - box_loss_scale

        # 计算中心偏移情况的loss，使用BCELoss效果好一些
        loss_x = torch.sum(self.BCELoss(x, y_true[..., 0]) * box_loss_scale * y_true[..., 4])
        loss_y = torch.sum(self.BCELoss(y, y_true[..., 1]) * box_loss_scale * y_true[..., 4])
        # 计算宽高调整值的loss
        loss_w = torch.sum(self.MSELoss(w, y_true[..., 2]) * 0.5 * box_loss_scale * y_true[..., 4])
        loss_h = torch.sum(self.MSELoss(h, y_true[..., 3]) * 0.5 * box_loss_scale * y_true[..., 4])
        # 计算置信度的loss
        loss_conf = torch.sum(self.BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + \
                    torch.sum(self.BCELoss(conf, y_true[..., 4]) * noobj_mask)
        # 计算类别损失
        loss_cls = torch.sum(self.BCELoss(pred_cls[y_true[..., 4] == 1], y_true[..., 5:][y_true[..., 4] == 1]))
        # 计算总损失
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        return loss, num_pos

    def get_target(self, l, targets, anchors, input_height, input_width):
        # 计算一共有多少张图片
        batch_size = len(targets)
        # 用于选取哪些先验框不包含物体
        noobj_mask = torch.ones(batch_size, len(self.anchors_mask[l]), input_height, input_width, requires_grad=False)
        # 让网络更加去关注小目标
        box_loss_scale = torch.zeros(batch_size, len(self.anchors_mask[l]), input_height, input_width,
                                     requires_grad=False)
        # batch_size, 3, 13, 13, 5 + num_classes
        y_true = torch.zeros(batch_size, len(self.anchors_mask[l]), input_height, input_width, self.bbox_attrs,
                             requires_grad=False)
        for b in range(batch_size):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])
            # 计算出正样本在特征层上的中心点
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * input_width
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * input_height
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()

            # 将真实框转换一个形式 (num_true_box, 4)
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            # 将先验框转换一个形式 (9, 4)
            anchor_shapes = torch.FloatTensor(
                torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            # 计算交并比
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)  # [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                # 判断这个先验框是当前特征点的哪一个先验框
                k = self.anchors_mask[l].index(best_n)
                # 获得真实框属于哪个网格点
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                # 取出真实框的种类
                c = batch_target[t, 4].long()

                # noobj_mask代表无目标的特征点
                noobj_mask[b, k, j, i] = 0
                # tx、ty代表中心调整参数的真实值
                y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1
                # 用于获得xywh的比例，大目标loss权重小，小目标loss权重大
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / input_width / input_height
            return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, input_height, input_width, noobj_mask):
        # 计算一共有多少张图片
        batch_size = len(targets)
        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(self.anchors_mask[l]), 1, 1).view(x.shape).to(self.device)  # (batch_size, 3, 13, 13)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(self.anchors_mask[l]), 1, 1).view(y.shape).to(self.device)

        # 按照网格格式生成先验框的宽高(batch_size, 3, 13, 13)
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]  # 生成先验框的宽高
        anchor_w = torch.FloatTensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).to(
            self.device)  # 沿着1轴选择index为0的元素
        anchor_h = torch.FloatTensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).to(self.device)
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(
            w.shape)  # (batch_size, 3, 13, 13)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # 计算调整后的先验框中心与宽高
        pred_boxes_x = torch.unsqueeze(x.data + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y.data + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w.data) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h.data) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(batch_size):
            # 将预测结果转换一个形式
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)  # (num_anchors, 4)
            # 计算真实框，并把真实框转换成相对于特征层的大小
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                # 计算出正样本在特征层上的中心点
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * input_width
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * input_height
                batch_target = batch_target[:, :4]
                # 计算交并比
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                #  每个先验框对应真实框的最大重合度
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask

    def calculate_iou(self, _box_a, _box_b):
        """
        计算交并比
        :param _box_a: 真实框(num_anchors, 4)，最后维度为x, y, w, h
        :param _box_b: 先验框(num_anchors, 4)
        :return:
        """
        # 计算真实框的左上角和右下角
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        # 计算先验框获得的预测框的左上角和右下角
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        # 将真实框和预测框都转化成左上角右下角的形式
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        # A为真实框的数量，B为先验框的数量
        A = box_a.size(0)
        B = box_b.size(0)

        # 计算交的面积
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        # 计算预测框和真实框各自的面积
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        # 求IOU
        union = area_a + area_b - inter
        return inter / union  # [A,B]

if __name__ == '__main__':
    from util import get_anchors

    anchors = get_anchors('../data/yolo_anchors.txt')[0]
    loss = YoloLoss(anchors, 3, [416, 416], 'cuda')
