# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict
from PIL import Image

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

color_map = [
    (0,0,255),
    (0,255,0),
    (255,0,0),
    (255,255,0),
    (32,240,160),
    (97,0,255),
    (255,255,0),
]

def draw_box_with_label(img, boxes, label, ids, points):
    box_num = len(boxes)
    label_map = {'clipper':0, 'grasper':1, 'irrigator':2}
    for i in range(box_num):

        id = ids[i]
        # cx, cy, w, h = boxes[i]
        # cv2.rectangle(img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (0,0,255), 2)
        # action = label_map[label.item()]
        # cv2.rectangle(img, (int(cx-w/2), int(cy-h/2)-35), (int(cx-w/2)+len(action)*18,int(cy-h/2)), (0,0,255), -1)
        # cv2.putText(img, action, (int(cx-w/2), int(cy-h/2)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # print(boxes[i])
        
        l,t,r,d = boxes[i]
        action = label[i]
        color = label_map[action] #int(id)%7
        cv2.rectangle(img, (int(l), int(t)), (int(r), int(d)), color_map[color], 2)
        
        # action = 'Grasper'
        cv2.rectangle(img, (int(l), int(d)-35), (int(l)+len(action)*18,int(d)), color_map[color], -1)
        cv2.putText(img, action, (int(l), int(d)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        first = True
        for j in range(len(points[id])):
            # print(j)
            if first:
                first = False
                continue
            cv2.line(img, points[id][j-1], points[id][j], color=color_map[color], thickness=2)

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_box, draw_box, imgsz, evaluate, half, project, name, exist_ok= \
        opt.save_dir, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, opt.save_box, \
        opt.draw_box, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )
    # boxes
    box_dict = defaultdict(list)
    point_dict = defaultdict(list)
    frame_dict = defaultdict(list)
    input_file_name = source[-9:-4] ## INPUT FILE NAME
    
    # Initialize
    
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        # show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

                
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        if frame_idx == 0:
            fps, w, h = 30, im0s.shape[1], im0s.shape[0]
            save_path = opt.save_dir + '/output_{}.mp4'.format(input_file_name) ## SAVE PATH EDIT
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        # print("pred: {}, shape: {}".format(pred,pred[0].shape))
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # print("det: {}, shape: {}".format(det,det.shape))
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                # print(xywhs, confs, clss)
                im = Image.fromarray(im0, 'RGB')
                im.save("img_wo_bb.png")

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        # box_dict
                        box_dict[id].append(bboxes)
                        l,t,r,d = bboxes
                        coor = (int((l+r)/2), int((t+d)/2))
                        point_dict[id].append(coor)
                        frame_dict[id].append(frame_idx)

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        # annotator.box_label(bboxes, label, color=colors(c, True))

                    if draw_box:
                        # print(name)
                        # print([int(outputs[i][5]) for i in range(len(outputs))])
                        labels = [names[int(outputs[i][5])] for i in range(len(outputs))]
                        boxes = [outputs[i][0:4] for i in range(len(outputs))]
                        ids = [outputs[i][4] for i in range(len(outputs))]
                        # print(boxes)
                        # print(ids)
                        # Write MOT compliant results to file
                        draw_box_with_label(im0s, boxes, labels, ids, point_dict)
                        
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')
        vid_writer.write(im0s)
    vid_writer.release()  # release previous video writer
    # Save boxes
    if save_box:
        sec = 1
        h_w_ratio = 171/256

        while box_dict:
            for i in box_dict:
                if len(frame_dict[i]) < 30:
                    continue
                frame_count = 0
                vid_path, vid_writer = [None] * 1, [None] * 1
                # if not os.path.exists(opt.save_dir):
                #     os.mkdir(opt.save_dir)
                save_path = opt.save_dir + '/output_{}.mp4'.format(input_file_name) ## SAVE PATH EDIT
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                print(input_file_name, i, sec)
                
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
                    if frame_idx not in frame_dict[i]: # or frame_count>30:
                        continue
                    else:
                        l,t,r,d = box_dict[i][frame_count]
                        # print(l,t,r,d)
                        frame_count += 1
                        # l,t,r,d = max((l+r)//2-48,0),max((t+d)//2-64,0),min((l+r)//2+48,im0s.shape[1]),min((t+d)//2+64,im0s.shape[0])
                        frame_wid, frame_hei = r-l, d-t
                        l,t,r,d = max(0,l-frame_wid//4),max(0,t-frame_hei//4),min(im0s.shape[1],r+frame_wid//4),min(im0s.shape[0],d+frame_hei//4)
                        im_new = im0s[t:d,l:r,:]
                        ratio = (r-l)/(d-t)
                        if ratio < h_w_ratio:
                            new_w, new_h = round(256*ratio), 256
                            gap = 171 - new_w
                            im_new = cv2.resize(im_new, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            im_new = cv2.copyMakeBorder(im_new, 0,0,gap//2,gap-gap//2, cv2.BORDER_CONSTANT, value=[0,0,0])
                        else:
                            new_w, new_h = 171, round(171/ratio)
                            gap = 256 - new_h
                            im_new = cv2.resize(im_new, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            im_new = cv2.copyMakeBorder(im_new, gap//2, gap-gap//2,0,0,cv2.BORDER_CONSTANT, value=[0,0,0])

                        # im_new = cv2.resize(im_new, (128,256), interpolation=cv2.INTER_CUBIC)
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            fps, w, h = 30, im_new.shape[1], im_new.shape[0]
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im_new)
                print("Box id #{} saved to {}, total {} video(s) are saved!".format(i,save_path,sec))
            for i in list(box_dict.keys()):
                if len(box_dict[i]) < 30:
                    del box_dict[i]
                    del frame_dict[i]
                else:
                    box_dict[i] = box_dict[i][30:]
                    frame_dict[i] = frame_dict[i][30:]
            sec += 1


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_vid:
    #     print('Results saved to %s' % save_path)
    #     if platform == 'darwin':  # MacOS
    #         os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_dir', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',  help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-box', action='store_true', default=False, help='save video tracking results')
    parser.add_argument('--draw_box', action='store_true', default=True, help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="tool/deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
