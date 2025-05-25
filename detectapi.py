
from pathlib import Path
from torch.backends import cudnn

"""
使用面向对象编程中的类来封装，需要去除掉原始 detect.py 中的结果保存方法，重写
保存方法将结果保存到一个 csv 文件中并打上视频的对应帧率

"""


class YoloOpt:
    def __init__(self, weights='weights/last.pt',
                 imgsz=(640, 640), conf_thres=0.25,
                 iou_thres=0.45, device='cpu', view_img=False,
                 classes=None, agnostic_nms=False,
                 augment=False, update=False, exist_ok=False,
                 project='/detect/result', name='result_exp',
                 save_csv=True):
        self.weights = weights  # 权重文件地址
        self.source = None  # 待识别的图像
        if imgsz is None:
            self.imgsz = (640, 640)
        self.imgsz = imgsz  # 输入图片的大小，默认 (640,640)
        self.conf_thres = conf_thres  # object置信度阈值 默认0.25  用在nms中
        self.iou_thres = iou_thres  # 做nms的iou阈值 默认0.45   用在nms中
        self.device = device  # 执行代码的设备，使用半精度，封装成了GPU方法
        self.view_img = view_img  # 是否展示预测之后的图片或视频 默认False
        self.classes = classes  # 只保留一部分的类别，默认是全部保留
        self.agnostic_nms = agnostic_nms  # 进行NMS去除不同类别之间的框, 默认False
        self.augment = augment  # augmented inference TTA测试时增强/多尺度预测，可以提分
        self.update = update  # 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
        self.exist_ok = exist_ok  # 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
        self.project = project  # 保存测试日志的参数，本程序没有用到
        self.name = name  # 每次实验的名称，本程序也没有用到
        self.save_csv = save_csv  # 是否保存成 csv 文件，本程序目前也没有用到
        