import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model_object_detection(num_classes):
    # 加载在COCO上预训练的实例分割模型faster-rcnn
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', weights_backbone='DEFAULT')

    # 获取分类器的输入种类数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 将预测器头替换为新的
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == '__main__':
    model = get_model_object_detection(4)

    print(model)