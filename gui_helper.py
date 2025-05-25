from tkinter import *
from tkinter.filedialog import askdirectory,askopenfile
from tkinter import messagebox
from PIL import Image
import os
from time import time,sleep
import PIL.Image
import ttkbootstrap as ttk
import PIL
from threading import Thread
import cv2
import random
import sys
from pathlib import Path
import torch
import csv
from detectapi import  YoloOpt

#确保 YOLOv5 项目的根目录被添加到 Python 的模块搜索路径中，并且将根目录路径转换为相对于当前工作目录的相对路径，
# 这样无论项目在哪里被执行，都能够正确地找到项目中的模块。


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync

class DetectAPI:
    def __init__(self, weights, imgsz=640):
        global opted,photo_frame,total,classes
        self.opt = YoloOpt(weights=weights, imgsz=imgsz)
        weights = self.opt.weights
        imgsz = self.opt.imgsz        
        #用于记录识别进度
        self.counter = 0
        # Initialize 初始化
        # 获取设备 CPU/CUDA
        default_cuda_device = torch.cuda.current_device()
        self.device = select_device(self.opt.device) if not torch.cuda.is_available() else select_device(default_cuda_device)

        # 使用半精度，成功解决使用gpu推断问题
        self.half = self.device.type == default_cuda_device  # # FP16 supported on limited backends with CUDA

        # Load model 加载模型
        self.model = DetectMultiBackend(weights, self.device, dnn=False)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.detected = set()

        
        self.video_fps = 0
        self.video_time = 0
        self.frame_index = 0
        self.saving_video = False

        # 不使用半精度
        if self.half:
            self.model.half() # switch to FP16

        # read names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.is_video = False
        print(self.names)
    def reset(self):
        self.counter = 0
        opted.set(self.counter)
        self.detected.clear()
        classes.set(0)
        self.frame_index = 0
        self.video_time = 0
        self.video_fps = 0

        

        #self.result 的元素是列表，一个列表代表一张图片的识别结果，
        # 每个列表的元素是元组，每个元组代表一张图片里面的一个识别结果

    def multidetect(self,source):
        
        # 输入 detect([img])
        self.reset()            #初始化计数器
        #print(self.names)
        if type(source) != list:
            raise TypeError('source must a list and contain picture read by cv2')
        # DataLoader 加载数据
        # 直接从 source 加载数据，可以一次性加载多张，不过要传入以cv2对象为元素的列表
        dataset = LoadImages(source)
        # 源程序通过路径加载数据，现在 source 就是加载好的数据，因此 LoadImages 就要重写
        bs = 1 # set batch size
        
        # Run inference。预热模型
        result = []
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        dt, seen = (Profile(), Profile(), Profile()), 0

        
        total.set(len(source))
        config_pgb()
        #多线程提示
        start_tip()
        
        timer = Timer()
        for im, im0s in dataset:
            timer.start()
            with dt[0]:
                #将np数组转化成张量(cv2对象是np数组)
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # Inference
                
                pred = self.model(im, augment=self.opt.augment)[0]

                # NMS，定制最大识别数量
                with dt[2]:
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, max_det=20)

                # Process predictions
                # 处理每一张图片
                det = pred[0]  
                im0 = im0s.copy()  # copy 一个原图片的副本图片
                result_txt = []  # 储存检测结果，每新检测出一个物品，长度就加一。
                                 # 每一个元素是列表形式，储存着 类别，坐标，置信度
                # 设置图片上绘制框的粗细，类别名称
                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 映射预测信息到原图
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    #
                    for *xyxy, conf, cls in reversed(det):
                        line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                        result_txt.append(line)
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])
                
                timer.end()
                result.append((im0, result_txt,round(timer.ed-timer.st,2)))        # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
                #记录这张图片识别到的对象
                for resultofimg in result_txt:
                    self.detected.add(resultofimg[0])
                #更新识别到的种类

                classes.set(len(self.detected))
                self.counter +=1
                opted.set(self.counter)
                
                config_pgb()
                config_meter()
                load_photo(photo_frame,im0)             #显示识别后的图像
                if opted.get() == total.get():
                    sleep(0.5)
        return result, self.names
    
    def detect_a_photo(self, source):   
        # 输入 detect([img])
        self.reset()            #初始化计数器
        totalclasses.set(len(self.names))
        config_meter_universial(self)
   
        
        if type(source) != list:
            raise TypeError('source must a list and contain picture read by cv2')
        # DataLoader 加载数据
        # 直接从 source 加载数据，可以一次性加载多张，不过要传入以cv2对象为元素的列表
        dataset = LoadImages(source)
        # 源程序通过路径加载数据，现在 source 就是加载好的数据，因此 LoadImages 就要重写
        bs = 1 # set batch size
        # 保存的路径
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference。预热模型
        result = []
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        dt, seen = (Profile(), Profile(), Profile()), 0

        for im, im0s in dataset:
            with dt[0]:
                #将np数组转化成张量(cv2对象是np数组)
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # Inference
                if not self.saving_video:
                    start_tip()

                pred = self.model(im, augment=self.opt.augment)[0]

                # NMS，定制最大识别数量
                with dt[2]:
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, max_det=20)

                # Process predictions
                # 处理每一张图片
                det = pred[0]  # API 一次只处理一张图片，因此不需要 for 循环
                im0 = im0s.copy()  # copy 一个原图片的副本图片
                result_txt = []  # 储存检测结果，每新检测出一个物品，长度就加一。
                                 # 每一个元素是列表形式，储存着 类别，坐标，置信度
                # 设置图片上绘制框的粗细，类别名称
                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 映射预测信息到原图
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                        result_txt.append(line)
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])
                result.append((im0, result_txt))  # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
                #记录这张图片识别到的对象
                for resultofimg in result_txt:
                    self.detected.add(resultofimg[0])
                #更新识别到的种类

                classes.set(len(self.detected))
                config_meter_universial(self)
            return result, self.names

    def prepare_to_save_video(self):
        self.saving_video = True
        if save_state_val.get() == 'yes':  
            counter = 0                                      #记录用于增量存储的目录序号                   
            exist_dirs = parse_dirofdir(os.path.join(os.getcwd(),'runs\\'))         
            exist_dir = [os.path.basename(dir).replace('exp','') for dir in exist_dirs]
            exist_counter = []
            for dir in exist_dir:
                if dir.isdigit():
                    exist_counter.append(int(dir))
            if  exist_counter:
                counter = max(exist_counter)+1              #增量存储
            label = f'exp{counter}'          
            saved_dir = os.path.join(os.path.join(os.getcwd(),'runs\\'), label)
            os.makedirs(saved_dir)                          #创建存储目录
            filename = os.path.basename(src.get())
            filename = os.path.splitext(filename)[0]+'.mp4'
            saved_path = os.path.join(saved_dir,filename)
            target_video = cv2.VideoCapture(src.get())                          #创建视频对象
            fps = target_video.get(cv2.CAP_PROP_FPS)                               #获取视频帧率
            size = (int(target_video.get(cv2.CAP_PROP_FRAME_WIDTH)),               #获取视频的高度和宽度
                    int(target_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if not target_video.isOpened():                                        #若视频没有成功加载并打开，退出程序
                print("打开失败")
                return None,None,None
            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            fourcc_avi = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')             #获取.avi格式的视频编码
            out = cv2.VideoWriter(saved_path,fourcc_mp4, fps, size) 
            return target_video,out,saved_dir
        else:
            return None,None,None


    

    def detect_video(self,source):
        target_video,out,saved_dir = self.prepare_to_save_video()
        # 输入 detect([img])
        if type(source) != list:
            raise TypeError('source must a list and contain picture read by cv2')
        # DataLoader 加载数据
        # 直接从 source 加载数据，可以一次性加载多张，不过要传入以cv2对象为元素的列表
        dataset = LoadImages(source)
        # 源程序通过路径加载数据，现在 source 就是加载好的数据，因此 LoadImages 就要重写
        bs = 1 # set batch size        
        # Run inference。预热模型
        result = []
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        dt, seen = (Profile(), Profile(), Profile()), 0
        total.set(len(source))
        config_pgb()
        #多线程提示
        start_tip()
        for im, im0s in dataset:
            with dt[0]:
                #将np数组转化成张量(cv2对象是np数组)
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                # Inference             
                pred = self.model(im, augment=self.opt.augment)[0]
                # NMS，定制最大识别数量
                with dt[2]:
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, max_det=20)
                # Process predictions
                # 处理每一张图片
                det = pred[0]  
                im0 = im0s.copy()  # copy 一个原图片的副本图片
                result_txt = []  # 储存检测结果，每新检测出一个物品，长度就加一。
                                 # 每一个元素是列表形式，储存着 类别，坐标，置信度
                # 设置图片上绘制框的粗细，类别名称
                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 映射预测信息到原图
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    #
                    for *xyxy, conf, cls in reversed(det):
                        line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                        result_txt.append(line)
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])
                        # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
                self.frame_index += 1
                #图片实时显示放在另一个线程
                t3 = Thread(target=load_photo,args=(photo_frame,im0))
                t3.start()             #显示识别后的图像
                t3.join()
            
                result.append((im0, result_txt))
                
                self.counter +=1
                opted.set(self.counter)
                config_video_pgb(self.video_time,round(self.frame_index/self.video_fps,2))
                if out:
                    out.write(im0)
        if out:
            target_video.release()
            out.release()
        self.reset()            #初始化计数器


        return result, self.names
    

    def detect(self, source,live = False):   
        # 输入 detect([img])
        if type(source) != list:
            raise TypeError('source must a list and contain picture read by cv2')
        # DataLoader 加载数据
        # 直接从 source 加载数据，可以一次性加载多张，不过要传入以cv2对象为元素的列表
        dataset = LoadImages(source)
        # 源程序通过路径加载数据，现在 source 就是加载好的数据，因此 LoadImages 就要重写
        bs = 1 # set batch size
        # 保存的路径
        vid_path, vid_writer = [None] * bs, [None] * bs
        # Run inference。预热模型
        result = []
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        dt, seen = (Profile(), Profile(), Profile()), 0

        for im, im0s in dataset:
            with dt[0]:
                #将np数组转化成张量(cv2对象是np数组)
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # Inference
                if not self.saving_video and not live:
                    start_tip()

                pred = self.model(im, augment=self.opt.augment)[0]

                # NMS，定制最大识别数量
                with dt[2]:
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, max_det=20)

                # Process predictions
                # 处理每一张图片
                det = pred[0]  # API 一次只处理一张图片，因此不需要 for 循环
                im0 = im0s.copy()  # copy 一个原图片的副本图片
                result_txt = []  # 储存检测结果，每新检测出一个物品，长度就加一。
                                 # 每一个元素是列表形式，储存着 类别，坐标，置信度
                # 设置图片上绘制框的粗细，类别名称
                annotator = Annotator(im0, line_width=3, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 映射预测信息到原图
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                        result_txt.append(line)
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])
                result.append((im0, result_txt))  # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
            return result, self.names



#计时器
class Timer:
    def __init__(self):
        self.st = None
        self.ed =  None

    def start(self):
        st = time()
        self.st = st

    def end(self):
        ed = time()
        self.ed = ed



def start_message():
    global root
    tl = ttk.Toplevel(title='-',size = (320,125),resizable=(False, False))
    x_center = root.winfo_x() + (root.winfo_width() // 2)
    y_center = root.winfo_y() + (root.winfo_height() // 2)
    tl.geometry("+{}+{}".format(x_center, y_center))
    label3 = ttk.Label(tl, text="Recognition \n      start!",bootstyle = 'inverse-warning',font = ('Arial',20,'bold','italic'))
    label3.pack()
    tl.after(1000,tl.destroy)


def start_tip():
    t1 = Thread(target=start_message)
    t1.start()
    

def create_window():
    root = ttk.Window(themename='vapor',title='recognizer_v2.0',size=(1950,1200),resizable=(False, False))
    return root        

def create_menu(root):
    Menubar = ttk.Menu(root)
    tipsmenu = ttk.Menu(Menubar)
    helpmenu = ttk.Menu(Menubar)
    Menubar.add_cascade(label='Announcements',menu=tipsmenu)
    tipsmenu.add_command(label='Tips about videos', command=lambda: messagebox.showinfo(title='info',message='We have a lightweight class video function only availible for .mp4 and .avi.\nVideos lasting for more than a minute would not be not recommended.'))
    tipsmenu.add_command(label='About Meter widget', command=lambda: messagebox.showwarning(title='info',message='Meter widget is not availible when operating videos or a single file'))
    Menubar.add_cascade(label='help',menu=helpmenu)
    helpmenu.add_command(label='shortcut keys', command=lambda: messagebox.showinfo(title='info',message='Left_arrow : Jump to previous result \n\n Right_arrow : Jump to next result \n\n Start: ctrl+s'))
    root.config(menu = Menubar)

def create_pgb(root):
    global total
    global opted
    pgb = ttk.Progressbar(root,length=822,value=0,bootstyle='success-striped')
    pgb.place(x=1050,y=500)
    pgb_label = ttk.Button(root,text='Progress',bootstyle='success-outline',width=7)
    
    pgb_label.place(x=900,y=490)
    pgb_text = ttk.Label(root,text=f'Finished: {opted.get()} / NaN',bootstyle='inverse-success',font=('Arial',10,'bold'))
    pgb_text.place(x=1050,y=463)
    return pgb,pgb_text,pgb_label
        
def config_pgb():
    global total
    global pgb
    global opted
    global pgb_label,src_dir,src
    #没选取目标目录的时候不改变初始值
    if src_dir.get() == '':
        return
    
    pgb.config(value=opted.get(),maximum=total.get())
    pgb_label.config(text=f'Finished: {opted.get()} / {total.get()}')

def config_video_pgb(video_time = 0,opted_time = 0):
    global total
    global pgb
    global opted
    global pgb_label,src
    #没选取目标目录的时候不改变初始值
    video_time = round(video_time,2)
    if src.get() == '':
        return
    
    if '.mp4' in os.path.basename(src.get()) or '.avi' in os.path.basename(src.get()) or '.wav' in os.path.basename(src.get()):
        pgb.config(value=opted_time,maximum=video_time)
        pgb_label.config(text=f'Finished: {opted_time}s / {video_time}s')
    
    
    
def create_file_loder(root,src=None,src_dir=None,weight=None):
    dir_entry_label = ttk.Label(root,text='Enter path of your photo directory',bootstyle='inverse-secondary')
    dir_entry_label.place(x=1050,y=263)
    dir_entry = ttk.Entry(root,width=58,bootstyle='secondary',font=('Times New Roman',10,'bold'),textvariable=src_dir)
    dir_entry.place(x=1050,y=300)
    dir_browse_btn = ttk.Button(root,text='Browse',width=7,bootstyle='secondary-outline',command=load_src_dir)
    dir_browse_btn.place(x=900,y=300)
    path_entry_label = ttk.Label(root,text='Enter path of your file(a single photo or video)',bootstyle='inverse-primary')
    path_entry_label.place(x=1050,y=163)
    path_entry = ttk.Entry(root,width=58,bootstyle='primary',font=('Times New Roman',10,'bold'),textvariable=src)
    path_entry.place(x=1050,y=200)
    browse_btn = ttk.Button(root,text='Browse',width=7,bootstyle='primary-outline',command=load_src)
    browse_btn.place(x=900,y=200)
    weight_entry = ttk.Entry(root,width=58,bootstyle='info',font=('Times New Roman',10,'bold'),textvariable=weight,state='disabled')
    weight_entry.place(x=1050,y=400)
    weight_entry_label = ttk.Label(root,text='Enter path of your weight',bootstyle='inverse-info')
    weight_entry_label.place(x=1050,y=363)
    weight_browse_btn = ttk.Button(root,text='Browse',width=7,bootstyle='info-outline',command=load_weight_path,state='disabled')
    weight_browse_btn.place(x=900,y=400)
    return dir_entry,dir_browse_btn,path_entry,browse_btn,weight_entry,weight_browse_btn


def load_photo_from_dir(photo_frame,img_list):
    global index,src_dir
    img = img_list[index.get()]
    if img is None:
        messagebox.showerror('error',"Encountered with a bad photo when loading the cover!(or not a photo!)")
        src_dir.set('')
        return  # 或者抛出异常
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换BGR到RGB
    img = cv2.resize(img,(750,750))
    pil_img = Image.fromarray(img)
    # 创建一个与tkinter兼容的图像对象
    photo = PIL.ImageTk.PhotoImage(pil_img)
    children = photo_frame.winfo_children()

    if not children:
        photo_label = ttk.Label(photo_frame, image=photo)
        photo_label.image = photo
        photo_label.pack()
    else:
        #不清除原来的子控件，节省响应时间
        photo_label = children[0]
        photo_label.config(image=photo)  # 重新设置photo_label的image属性
        photo_label.image = photo  # 保持对photo的引用

def load_src_dir():
    global src_dir,photo_frame,src
    global opted,index,meter,pgb_label,pgb,multidetect,total
    path = askdirectory()
    src_dir.set(path)
    try:
        files = parse_fileofdir(path)
        
        imgs= [cv2.imread(file) for file in files]
    except FileNotFoundError:
        pass
    src.set('')                             #更換文件路徑后清楚另一個路徑輸入段的輸入
    if not src_dir.get() == '':
        total.set(len(imgs))
    reset_pgb()
    config_meter()
    meter['textright'] = '/NaN'
    
    #更换识别目录后回到封面
    back_to_cover()
    if src_dir.get() != '':
        multidetect.photo_list = imgs
        load_photo_from_dir(photo_frame,imgs)
        

def reset_pgb():
    global pgb,pgb_label
    reset_data()                            #更换文件夹后index重设,pgb重设
    pgb_label.config(text=f'Finished: {opted.get()} / NaN')
    pgb.config(value=opted.get())

def load_src():
    global src,src_dir
    global photo_frame
    file = askopenfile()
    try:
        path = os.path.abspath(file.name)
        src.set(path)
                         #更換文件路徑后清楚另一個路徑輸入段的輸入
        #这里漏检查了一点：多了一句img = PIL.Image.open(path)
        if 'jpg' in os.path.basename(path) or 'png' in os.path.basename(path) or 'jpeg' in os.os.path.basename(path) or 'bmp' in os.os.path.basename(path):
            img  =  cv2.imread(path)
            load_photo(photo_frame,img)
    except AttributeError:
        pass
    reset_data()                            #更换文件夹后index重设,pgb重设
    pgb_label.config(text=f'Finished: {opted.get()} / NaN')
    pgb.config(value=opted.get())
    config_meter()
    meter['textright'] = '/NaN'
    src_dir.set('')

def load_weight_path():
    global classes,meter,custom_state_val
    global weight_path,weight_entry
    
    #获取状态值
    if custom_state_val.get() == 'yes':
        
        file = askopenfile()
        try:
            path = os.path.abspath(file.name)
            weight_path.set(path)
        except AttributeError:
            pass
        reset_data()
        config_meter()
        meter['textright'] = '/NaN'
        config_pgb()

        #更换权重后回到封面
        back_to_cover()

def save_tips():
    global root
    global savebtn
    global save_state_val
    par_path = os.getcwd()
    save_path = os.path.join(par_path,'runs\\')
    if save_state_val.get() == 'yes':
        response = messagebox.askokcancel(message=f'Results saved at\n{save_path}',title='Tips')
        if not response:
            save_state_val.set('no')

def custom_weight():
    global custombtn
    global weight_entry,weight_browse_btn
    global weight_path
    if custom_state_val.get() == 'yes':
        result = messagebox.askyesno(title = 'Question',message='Sure to choose your own weight?')
        if result:
            weight_entry.config(state='normal')
            weight_browse_btn.config(state='normal')
            weight_entry.delete(0, ttk.END)
        #回答否的时候相当于button没被点击
        else:
            custom_state_val.set('no')
            weight_browse_btn.config(state='disabled')
            weight_entry.config(state='disabled')

    elif custom_state_val.get() == 'no':
        weight_browse_btn.config(state='disabled')
        weight_entry.config(state='disabled')
        weight_path.set(r'.\weights\yolov5s.pt')


def create_btn(root):
    global save_state_val
    global custom_state_val
    start_btn = ttk.Button(root,text='Start',width=52,bootstyle='light-toolbutton',command=run_long_func,cursor='star')
    start_btn.place(x=30,y=1050)
    video_btn = ttk.Checkbutton(root,text='Live mode',width=68,bootstyle="danger-outline-toolbutton",command=run_long_func_video,
                                onvalue='yes',offvalue='no',variable=live_state_val,cursor='exchange')
    video_btn.place(x=900,y=680)
    save_btn_frame = ttk.Frame(root)
    save_btn = ttk.Checkbutton(save_btn_frame,bootstyle="square-toggle-success",text='  Save',command=save_tips,
    onvalue='yes',offvalue='no',variable=save_state_val)
    save_btn.pack()
    save_btn_frame.place(x=1750,y=590)
    custom_btn = ttk.Checkbutton(root,text='  Custom weight',bootstyle ='square-toggle-danger',command=custom_weight,
    onvalue='yes',offvalue='no',variable=custom_state_val)
    custom_btn.place(x=1450,y=590)
    return start_btn,save_btn,custom_btn,video_btn


#这里错就错在重新创建了一个label
# 传入cv2对象
def load_photo(photo_frame,img): 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换BGR到RGB
    img = cv2.resize(img,(750,750))
    pil_img = Image.fromarray(img)
    # 创建一个与tkinter兼容的图像对象
    photo = PIL.ImageTk.PhotoImage(pil_img)
    children = photo_frame.winfo_children()
    if not children:
        photo_label = ttk.Label(photo_frame, image=photo)
        photo_label.image = photo
        photo_label.pack()
    else:
        #不清除原来的子控件，节省响应时间
        photo_label = children[0]
        photo_label.config(image=photo)  # 重新设置photo_label的image属性
        photo_label.image = photo  # 保持对photo的引用
        

def create_combobox(root):
    global combox_val
    cb_label = ttk.Label(root,text='Avalible cameras',bootstyle='inverse-light')
    cb_label.place(x=1050,y= 560)
    cb = ttk.Combobox(root,bootstyle = 'light',state='readonly',textvariable=combox_val)
    cb.place(x=1050,y=590)
    box_dict = fill_combobox(cb)
    original_image = Image.open('./attachments/camera.png')
    
    # 调整图像大小
    resized_image = original_image.resize((80,80), Image.LANCZOS)
    icon = PIL.ImageTk.PhotoImage(resized_image)
    
    icon_label = ttk.Label(root,image=icon,bootstyle='inverse-light')
    icon_label.image = icon
    icon_label.place(x=920,y=560)
    return cb,box_dict,icon_label

def fill_combobox(box):
    val = get_windows_available_cameras()
    key = []
    for i in range(len(val)):
        if i == 0:
            key.append('defult camera (0)')
        else:
            key.append(f'camera ({i})')
    box['value'] = key
    box_dict = dict(zip(key,val))
    return box_dict
    

def create_photo_frame(root):
    global multidetect
    photo_frame = ttk.Frame(root,height=800,width=750) 
    #原来下面按钮之间有填充就是因为frame的样式设计成了light
    photo_frame.place(x=30,y=200)
    img = cv2.imread('./attachments/YOLO.png')
    load_photo(photo_frame,img)
    # next_btn = ttk.Button(photo_frame,text='Next',bootstyle='success-outline',width=25,command=multidetect.add_index)
    # back_btn = ttk.Button(photo_frame,text='Back',bootstyle='warning-outline',width=25,command=multidetect.reduce_index)
    # next_btn.pack(side=RIGHT)
    # back_btn.pack(side=LEFT)
    return photo_frame


def create_meter(root):
    meter = ttk.Meter(root,bootstyle="danger",subtextstyle="danger",showtext=True,
    subtext='Classes detected',metersize=200,amountused=0,stripethickness=5,textright = '/NaN')
    meter.place(x=1100,y=760)
    return meter

def config_meter():
    global meter,classes,totalclasses,multidetect
    
    try:#防止没选取权重并开始执行的时候这里会报错
        if 'ultraman' in multidetect.names.values():    
            if totalclasses.get() != 0:
                meter['amounttotal'] = totalclasses.get() - 15
                meter['textright'] = f'/{totalclasses.get() - 15}'
            meter['amountused'] = classes.get()
        else:
            if totalclasses.get() != 0:
                meter['amounttotal'] = totalclasses.get()
                meter['textright'] = f'/{totalclasses.get()}'
            meter['amountused'] = classes.get()
    except AttributeError:
        pass

def config_meter_universial(model):
    global meter,classes,totalclasses
    
    try:#防止没选取权重并开始执行的时候这里会报错
        if 'ultraman' in model.names.values():    
            if totalclasses.get() != 0:
                meter['amounttotal'] = totalclasses.get() - 15
                meter['textright'] = f'/{totalclasses.get() - 15}'
            meter['amountused'] = classes.get()
        else:
            if totalclasses.get() != 0:
                meter['amounttotal'] = totalclasses.get()
                meter['textright'] = f'/{totalclasses.get()}'
            meter['amountused'] = classes.get()
    except AttributeError:
        pass

def is_image_file(file_path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions
       


#获取目录下所有图片的地址
def parse_fileofdir(dir):
    files = os.listdir(dir)
    return [os.path.join(dir, f) for f in files if is_image_file(os.path.join(dir, f))]


def parse_dirofdir(dir):
    files = os.listdir(dir)
    return [os.path.join(dir, f) for f in files if os.path.isdir(os.path.join(dir, f))]

def parse_video():
        global src
        imgs = []
        video = cv2.VideoCapture(src.get())
        if not video.isOpened():
            messagebox.ERROR(title = 'Error', message = 'Failed to open the video')
        else:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            video_time = total_frames/fps
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                imgs.append(frame)
        return imgs,video_time,fps

#要修改api，使之可以识别多个对象
class multidetector():
    def __init__(self):
        global total,totalclasses
        global photo_frame
        global src_dir
        global src
        global weight_path
        global opted,classes,save_state_val
        self.opted_photo = None
        self.photo_list = None
        self.names = None
        self.results = None
        self.per_time = None

        self.video_frames = 0
        #解析视频用的属性，在这个类中不用到
        self.video_time = 0
        self.fps = 0

        self.save_path =  os.path.join(os.getcwd(),'runs\\')
        os.makedirs(self.save_path,exist_ok=True)
    
    

    def detect_video(self):
        self.reset()
        save_state_val.set('yes')                               #便于调适
        a = DetectAPI(weights=weight_path.get())
        a.is_video = True
        
        timer = Timer()
        timer.start()
        imgs,video_time,fps = parse_video()

        a.video_time = video_time
        a.video_fps = fps
        self.video_frames = len(imgs)
        config_video_pgb(video_time)
       
        with torch.no_grad():
            result,names = a.detect_video(imgs)
        timer.end()
        self.results = [r[1] for r in result]
        consume = round(timer.ed-timer.st,2)
        saved_dir = self.save_csv_video_results(a)
        
        #识别完变回封面
        back_to_cover()
        end_tips(consume,saved_dir)
        save_state_val.set('no')


    def save_csv_video_results(self,api):
        api.saving_video = True
        if save_state_val.get() == 'yes':  
            counter = 0                                      #记录用于增量存储的目录序号                   
            exist_dirs = parse_dirofdir(self.save_path)         
            exist_dir = [os.path.basename(dir).replace('exp','') for dir in exist_dirs]
            exist_counter = []
            for dir in exist_dir:
                if dir.isdigit():
                    exist_counter.append(int(dir))
            if  exist_counter:
                counter = max(exist_counter)              #不需增量存储
            label = f'exp{counter}'          
            saved_dir = os.path.join(self.save_path, label)
            os.makedirs(saved_dir,exist_ok=True)                          #创建存储目录
            filename = os.path.basename(src.get())
            filename = os.path.splitext(filename)[0]+'.mp4'
            
            #储存csv文件,当视频较短时才储存csv文件
            if self.video_frames <= 3000:
                csv_path = os.path.join(saved_dir,'result.csv')
                
                headers = ['frame_index','obj_classes','total_detected']
                with open(csv_path,'w',encoding='utf-8') as f:
                    writer = csv.DictWriter(f,fieldnames=headers)
                    writer.writeheader()
                    for result,index in zip(self.results,range(len(self.results))):
                        detected_obj = []
                        for obj in result:
                            info = f'{api.names[obj[0]]}:{obj[2]}'
                            detected_obj.append(info)
                        total_detected = len(detected_obj)
                        writer.writerow({'frame_index':f'{index}','obj_classes':detected_obj,
                        'total_detected':total_detected})
            print(f'Results saved to {saved_dir}')
            return saved_dir
        else:
            return None

    def save_results(self, results_pics):
        
        if save_state_val.get() == 'yes':  
            counter = 0                                      #记录用于增量存储的目录序号                   
            exist_dirs = parse_dirofdir(self.save_path)
            
            exist_dir = [os.path.basename(dir).replace('exp','') for dir in exist_dirs]
            exist_counter = []
            for dir in exist_dir:
                if dir.isdigit():
                    exist_counter.append(int(dir))
            if  exist_counter:
                counter = max(exist_counter)+1              

            label = f'exp{counter}'
            
            saved_dir = os.path.join(self.save_path, label)
            os.makedirs(saved_dir)                          #创建存储目录
            file_names = parse_fileofdir(src_dir.get())

            for pic,file_name in zip(results_pics, file_names):
                cv2.imwrite(os.path.join(saved_dir,os.path.basename(file_name)),pic)

            #储存csv文件
            csv_path = os.path.join(saved_dir,'result.csv')
            
            headers = ['basename','obj_classes','total_detected','time_consumed']
            with open(csv_path,'w',encoding='utf-8') as f:
                writer = csv.DictWriter(f,fieldnames=headers)
                writer.writeheader()
                for result,file_name,per_time in zip(self.results,file_names,self.per_time):
                    detected_obj = []
                    for obj in result:
                        info = f'{self.names[obj[0]]}:{obj[2]}'
                        detected_obj.append(info)
                    total_detected = len(detected_obj)
                    writer.writerow({'basename':os.path.basename(file_name),'obj_classes':detected_obj,
                    'total_detected':total_detected,'time_consumed':per_time})

            print(f'results saved to {saved_dir}')
            return saved_dir
        else:
            return None

    def detect(self):
        
        self.reset()
        a = DetectAPI(weights=weight_path.get())
        timer = Timer()
        timer.start()
        self.names = a.names
        totalclasses.set(len(self.names))
        config_meter()
        files = parse_fileofdir(src_dir.get())
        imgs = [cv2.imread(f) for f in files]
        with torch.no_grad():
            result,names = a.multidetect(imgs)
        timer.end()
        consume = round(timer.ed-timer.st,2)
        opted_imgs = [r[0] for r in result]
        self.results = [r[1] for r in result]
        self.per_time = [r[2] for r in result]
        self.opted_photo = opted_imgs
        load_photo_from_dir(photo_frame,opted_imgs)

        saved_dir = self.save_results(opted_imgs)
        end_tips(consume,saved_dir)


    def reset(self):
        self.names = None
        self.kinds = 0
        self.result = None
        self.detected = None
        self.video_frames =  0

    def add_index(self):
        global index
        if self.opted_photo:
            if index.get() >= total.get()-1:
                index.set(total.get()-1)  # 防止index超出
            else:
                index.set(index.get()+1)
            load_photo_from_dir(photo_frame,self.opted_photo)
        else:
            if index.get() >= total.get()-1:
                index.set(total.get()-1)  # 防止index超出
            else:
                index.set(index.get()+1)
            load_photo_from_dir(photo_frame,self.photo_list)
    def reduce_index(self):
        global index
        if self.opted_photo:
            if index.get() > 0:  # 防止index超出
                index.set(index.get()-1)
            load_photo_from_dir(photo_frame,self.opted_photo)
        else:
            if index.get() > 0:  # 防止index超出
                index.set(index.get()-1)
            load_photo_from_dir(photo_frame,self.photo_list)

def detectapi_groups():
    global multidetect
    multidetect.detect()
    
    

def save_sigle_result(source):
    global src
    save_path = os.path.join(os.getcwd(),'runs\\')
    if save_state_val.get() == 'yes':  
        counter = 0                                      #记录用于增量存储的目录序号                   
        exist_dirs = parse_dirofdir(save_path)
            
        exist_dir = [os.path.basename(dir).replace('exp','') for dir in exist_dirs]
        exist_counter = []
        for dir in exist_dir:
            if dir.isdigit():
                exist_counter.append(int(dir))
        if  exist_counter:
            counter = max(exist_counter)+1              #增量存储

        label = f'exp{counter}'
            
        saved_dir = os.path.join(save_path, label)
        os.makedirs(saved_dir)                          #创建存储目录
        filename = os.path.basename(src.get())
        cv2.imwrite(os.path.join(saved_dir,filename),source)
        print(f'results saved to {saved_dir}')
        return saved_dir

    else:
        return None
        

def detectapi_sigle():
    global photo_frame
    global src
    global weight_path
 
    a = DetectAPI(weights=weight_path.get())
    timer = Timer()
    timer.start()
    img = cv2.imread(src.get()) 
    with torch.no_grad():
        result,names = a.detect_a_photo([img])
        timer.end()
        consume = round(timer.ed-timer.st,2)
        #result的元素是元祖，元祖的第一个元素是画框后的图片
        img = result[0][0]  
        load_photo(photo_frame,img)
        saved_dir = save_sigle_result(img)
        end_tips(consume,saved_dir)





def detectapi():
    global src_dir
    global src,pgb_label,opted,meter
    global weight_path,pgb,classes,live_state_val,videobtn
    if live_state_val.get() == 'yes':                                               #实时模式下不做响应
        return
    operating_config('disabled')
    #程序流控制，首先判断是否两个都选上,处理源文件输入错误
    if src_dir.get() != '' and src.get() != '':
        messagebox.showerror('Error','Please choose either a file or a directory')
    elif src_dir.get() == '' and src.get() == '':
        messagebox.showerror('Error','Please choose a file or a directory')
    elif weight_path.get() == '':
        messagebox.showerror('Error','Please choose a weight file')

    elif src.get() != '':
        try:
            if '.mp4' in os.path.basename(src.get()) or '.avi' in os.path.basename(src.get()) or '.wav' in os.path.basename(src.get()):
                multidetect.detect_video()
            else:
                detectapi_sigle()
        except TypeError as e:
            messagebox.showerror('Error','Failed to load image or video')
            pgb_label['text'] = f'Finished: {opted.get()} / NaN'
            pgb['value'] = 0
            meter['textright'] = '/NaN'
            meter['amountused'] = 0
            print(e)
        # except AssertionError:
        #     messagebox.showerror('Error','Failed to load weight file')
        #     pgb_label['text'] = f'Finished: {opted.get()} / NaN'
        #     pgb['value'] = 0

    elif src_dir.get() != '':
        try:
            detectapi_groups()
        except TypeError:
            messagebox.showerror('Error','Failed to load image or video')
            pgb_label['text'] = f'Finished: {opted.get()} / NaN'
            meter['textright'] = '/NaN'
            meter['amountused'] = 0
            pgb['value'] = 0

        # except AssertionError:
        #     messagebox.showerror('Error','Failed to load weight file')
        #     pgb_label['text'] = f'Finished: {opted.get()} / NaN'
        #     pgb['value'] = 0
    operating_config('normal')


def end_tips(consume,saved_path=None):
    global save_state_val
    if save_state_val.get() == 'yes' and saved_path:
        messagebox.showinfo(title='Congratulations',message=f'Done! Elapsed time:  {consume} s\n\nResults saved to : {saved_path}')
    else:
        messagebox.showinfo(title='Congratulations',message=f'Done! Elapsed time:  {consume} s')


def back_to_cover():
    global photo_frame
    cover = cv2.imread('./attachments/YOLO.png')
    load_photo(photo_frame,cover)

def reset_data():
    #更换目录后把已识别的图片和已识别种类置零
    global opted,index,classes,totalclasses
    opted.set(0)
    index.set(0)
    classes.set(0)
    totalclasses.set(0)

def run_long_func(*args):
    global live_state_val
    if live_state_val.get() != 'yes':
        t = Thread(target=detectapi)
        t.daemon = True
        t.start()


def camera_not_exist():
    global live_state_val
    messagebox.showerror('Error','The selected camera is not avalible! \n\n We\'ve automatically updated the availible camera list!')
    box_dict = fill_combobox(cb)
    live_state_val.set('no')
    live_mode_config('normal')

def no_camera():
    global live_state_val
    messagebox.showerror('Error','Please choose a camera!')
    live_state_val.set('no')
    live_mode_config('normal')

def live_detect():
    global photo_frame
    global src
    global weight_path
    global live_state_val,save_state_val
    global stop_threads
    global combox_val,box_dict,cb
    if live_state_val.get() == 'yes':
        reset_pgb()
        save_state_val.set('no')        #实时模式下自动关闭保存
        live_mode_config('disabled')
        if combox_val.get() == '':
            no_camera()
            return
        
        stop_threads = False
        a = DetectAPI(weights=weight_path.get())
        cap = cv2.VideoCapture(box_dict[combox_val.get()])

        if not cap.isOpened():          #攝像頭索引不對（溢出或沒有這個攝像頭）
            camera_not_exist()
            return
        
        while not stop_threads:
            ret,frame = cap.read()
            if not ret:
                messagebox.showerror('Error','Failed to load live stream!')
                break
            with torch.no_grad():
                result,_ = a.detect([frame],live=True)
                #result的元素是元祖，元祖的第一个元素是画框后的图片
                img = result[0][0]  
                load_photo(photo_frame,img)
    else:
        live_mode_config('normal')
        stop_threads = True
        print("live mode overd!")
        sleep(0.5)                  #延迟响应，保证能够回到封面
        back_to_cover()
            
        

def live_mode_config(state):                              #视频模式下对控件进行状态限定
    global start_btn,icon_label,cb
    global savebtn
    global custombtn
    global custom_state_val
    global dir_entry,dir_browse_btn,path_entry,browse_btn,weight_entry,weight_browse_btn,pgbbtn
    start_btn.config(state=state)
    savebtn.config(state=state)
    custombtn.config(state=state)
    dir_entry.config(state=state)
    dir_browse_btn.config(state=state)
    path_entry.config(state = state)
    browse_btn.config(state = state)
    pgbbtn.config(state=state)
    icon_label.config(state=state)
    cb.config(state='disabled') if state == 'disabled' else cb.config(state='readonly')
    if custom_state_val.get() == 'yes' and state == 'normal':
        weight_entry.config(state = 'normal')
        weight_browse_btn.config(state='normal')
    else:
        weight_entry.config(state = 'disabled')
        weight_browse_btn.config(state='disabled')

def operating_config(state):
    global start_btn,cb,icon_label
    global savebtn,videobtn
    global custombtn
    global custom_state_val
    global dir_entry,dir_browse_btn,path_entry,browse_btn,weight_entry,weight_browse_btn,pgbbtn
    start_btn.config(state=state)
    savebtn.config(state=state)
    custombtn.config(state=state)
    dir_entry.config(state=state)
    dir_browse_btn.config(state=state)
    path_entry.config(state = state)
    browse_btn.config(state = state)
    videobtn.config(state=state)
    pgbbtn.config(state=state)
    icon_label.config(state=state)
    cb.config(state='disabled') if state == 'disabled' else cb.config(state='readonly')
    if custom_state_val.get() == 'yes' and state == 'normal':
        weight_entry.config(state = 'normal')
        weight_browse_btn.config(state='normal')
    else:
        weight_entry.config(state = 'disabled')
        weight_browse_btn.config(state='disabled')


def run_long_func_video(*args):
    t = Thread(target=live_detect)
    #守护线程，随主程序结束
    t.daemon = True
    t.start()



def get_windows_available_cameras():        #獲取可用的攝像頭列表
    try:
        available_cameras = []
        for index in range(5):  # 假设最多检查10个摄像头
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()
    except EXCEPTION as e:
        print(e)
    return available_cameras




if __name__ == '__main__':
    #创建主窗口        
    root = create_window()

    #############################################定义全局变量##########################################
    index = IntVar(value = 0)   #实现上下翻页的索引
    #模型一共有多少种种类
    classes = IntVar(value=0)
    totalclasses = IntVar(value=0)
    #待识别的总数
    opted = IntVar(value=0)
    total = IntVar(value=0)
    #给源目录定制一个变量
    src_dir = StringVar(value='')
    #给源文件的路径定制一个变量
    src =  StringVar(value='')
    #给权重路径定制一个变量
    weight_path = StringVar(value=r'.\weights\flower_best.pt')
    #给保存结果按钮定制一个变量
    save_state_val = StringVar(value='')
    custom_state_val = StringVar(value='')
    live_state_val = StringVar(value='')
    combox_val = StringVar(value='')
    stop_threads = False
    
    #############################################定义全局变量##########################################

    create_menu(root)
    multidetect = multidetector()               #創建多重識別器對象
    
    photo_frame = create_photo_frame(root)
    dir_entry,dir_browse_btn,path_entry,browse_btn,weight_entry,weight_browse_btn = create_file_loder(root,src,src_dir,weight_path)
    pgb,pgb_label,pgbbtn = create_pgb(root)
    meter = create_meter(root)
    
    start_btn,savebtn,custombtn,videobtn = create_btn(root)

    cb,box_dict,icon_label = create_combobox(root)

    #创建用于绑定键盘的动作的匿名函数,视频模式下该函数不起作用
    on_left_arrow_press = lambda event : multidetect.reduce_index() if live_state_val.get() != 'yes' else lambda:0
    on_right_arrow_press= lambda event : multidetect.add_index() if live_state_val.get() != 'yes' else lambda:0

    #事件绑定
    root.bind("<Left>", on_left_arrow_press)
    root.bind("<Right>", on_right_arrow_press)
    root.bind('<Control-s>',run_long_func)

    root.mainloop()

