# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    继承自 BasePredictor 类，用于基于检测模型进行预测的类。

    示例:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        print("\n DetectionPredictor类postprocess()被调用,返回一个 Results 对象列表。","地址ultralytics-main\\ultralytics\\models\\yolo\\detect.py")
        """对预测结果进行后处理，并返回一个 Results 对象列表。"""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
        )#preds:List[torch.Tensor]其.shape为(object_num,6),6为(x1,y1,x2,y2,score,class_id)

        if not isinstance(orig_imgs, list):  # 如果输入图像是 torch.Tensor 而不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        result = self.construct_results(preds, img, orig_imgs, **kwargs)
        return result

    def construct_results(self, preds, img, orig_imgs):
        print("\n DetectionPredictor类construct_results()被调用，从预测结果构建结果对象列表。","地址ultralytics-main\\ultralytics\\models\\yolo\\detect.py")
        """
        从预测结果构建结果对象列表。

        参数:
            preds (List[torch.Tensor]): 预测的边界框和分数的列表。
            img (torch.Tensor): 预处理后的图像。
            orig_imgs (List[np.ndarray]): 预处理前的原始图像列表。

        返回:
            (list): 包含原始图像、图像路径、类别名称和边界框的结果对象列表。
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        从预测结果构建结果对象。

        参数:
            pred (torch.Tensor): 预测的边界框和分数。
            img (torch.Tensor): 预处理后的图像。
            orig_img (np.ndarray): 预处理前的原始图像。
            img_path (str): 原始图像的路径。

        返回:
            (Results): 包含原始图像、图像路径、类别名称和边界框的结果对象。
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])