# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlpackage          # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import pathlib
import torch
import time
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

# Adjust pathlib for Windows compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "best.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(1024, 1024),  # inference size (height, width)
    conf_thres=0.5,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=True,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        ... [Docstring 생략]
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    # 클래스 이름을 키로 사용하는 매핑으로 수정
    class_price_mapping = {
        "스프라이트": 2000,
        "마데카솔": 7500,
        "동원고추참치": 4400,
        "농심신라면컵": 1500,
        "농심매운새우깡": 1700,
        "2080퓨어치약클린민트향": 3000
    }
    # 총 가격 변수 초기화 
    total_price = 0
    
    # 감지된 클래스와 시간을 추적하는 딕셔너리 
    detected_items = {}

    # 구매 항목을 저장할 리스트 초기화 
    purchased_items = []

    # 중앙 정렬 텍스트 박스 함수 정의
    def draw_centered_text_box(draw, box_position, text, font, text_color=(0, 0, 0), box_fill=(255, 255, 255), box_outline=(0, 0, 0), padding=10):
        """
        텍스트를 중앙에 배치한 사각형을 그리는 함수.
        """
        x, y = box_position
        # 텍스트의 경계 상자 계산
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 박스 크기 계산
        box_width = text_width + 2 * padding
        box_height = text_height + 2 * padding

        # 사각형 그리기
        draw.rectangle([x, y, x + box_width, y + box_height], fill=box_fill, outline=box_outline, width=2)

        # 텍스트 위치 계산 (중앙 정렬)
        text_x = x + (box_width - text_width) / 2 - bbox[0]
        text_y = y + (box_height - text_height) / 2 - bbox[1]

        # 텍스트 그리기
        draw.text((text_x, text_y), text, font=font, fill=text_color)

    # OpenCV 창 설정 (모든 플랫폼에서 단일 창 사용)
    if view_img:
        window_name = "YOLOv5 Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 창 크기 조정 가능하게 설정
        cv2.resizeWindow(window_name, 800, 600)  # 원하는 초기 창 크기로 설정 (가로 1024, 세로 768)

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="", encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    class_name = names[c]  # 클래스 이름
                    confidence = float(conf)

                    if save_csv:
                        label = class_name if hide_conf else f"{class_name}"
                        confidence_str = f"{confidence:.2f}"
                        write_to_csv(p.name, label, confidence_str)

                    current_time = time.time()

                    if class_name in detected_items:
                        # 감지된 시간 갱신
                        detected_items[class_name]['last_seen'] = current_time

                        # 2초 이상 지속적으로 감지되었는지 확인
                        time_detected = detected_items[class_name]['last_seen'] - detected_items[class_name]['first_detected']
                        if time_detected >= 2 and not detected_items[class_name]['added']:
                            total_price += class_price_mapping.get(class_name, 0)  # 총 가격 업데이트
                            detected_items[class_name]['added'] = True  # 가격이 추가되었음을 표시
                            purchased_items.append(f"{class_name}: {class_price_mapping.get(class_name, 0)}원")  # 구매 항목 추가
                    else:
                        # 새로운 상품이 감지된 경우 시간 기록
                        detected_items[class_name] = {
                            'first_detected': current_time,
                            'last_seen': current_time,
                            'added': False
                        }

                    if save_img or save_crop or view_img:  # Add bbox to image
                        if detected_items[class_name]['added']:
                            label = f"{class_name}: {class_price_mapping.get(class_name, 0)}원"
                        else:
                            label = class_name if hide_conf else f"{class_name} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # 객체 감지 후, 객체가 더 이상 감지되지 않은 경우 detected_items에서 제거
            current_time = time.time()
            remove_keys = []
            for class_name, info in detected_items.items():
                if current_time - info['last_seen'] > 5:
                    remove_keys.append(class_name)

            for key in remove_keys:
                del detected_items[key]

            # Stream results
            im0 = annotator.result()  # YOLOv5가 처리한 프레임 가져오기
            im0 = im0.copy()  # 읽기 전용 배열을 쓰기가 가능한 배열로 복사

            if view_img:
                # 폰트 경로 설정
                font_path_bold = "C:/Windows/Fonts/malgunbd.ttf"     # 볼드 폰트

                # 폰트 로드
                try:
                    font_large_bold = ImageFont.truetype(font_path_bold, 30)  # "상품을 스캔해주세요" 용 (크기: 30, 굵게)
                except IOError:
                    print(f"폰트 파일을 찾을 수 없습니다: {font_path_bold}")
                    font_large_bold = ImageFont.load_default()

                try:
                    font_bold = ImageFont.truetype(font_path_bold, 20)  # "총 가격" 및 구매 항목 용 (크기: 20, 굵게)
                except IOError:
                    print(f"폰트 파일을 찾을 수 없습니다: {font_path_bold}")
                    font_bold = ImageFont.load_default()

                # 중앙 정렬 텍스트 박스 함수 정의 (중복 방지 위해 함수는 상단에 정의)
                # draw_centered_text_box 함수는 이미 상단에서 정의되어 있습니다.

                # OpenCV 이미지를 Pillow 이미지로 변환
                im_pil = Image.fromarray(im0)
                draw = ImageDraw.Draw(im_pil)

                # "상품을 스캔해주세요" 텍스트와 박스 그리기
                text_main = "상품을 스캔해주세요"
                # 박스 위치 설정 (왼쪽 상단에 약간의 여백)
                box_main_x, box_main_y = 10, 10
                draw_centered_text_box(draw, (box_main_x, box_main_y), text_main, font_large_bold, box_fill=(255, 255, 255), box_outline=(0, 0, 0), padding=15)

                # "총 가격" 텍스트와 박스 위치 계산 (오른쪽 하단)
                total_price_text = f"총 가격: {total_price}원"
                bbox_total = draw.textbbox((0, 0), total_price_text, font=font_bold)  # 굵은 폰트로 경계 상자 계산
                total_price_width, total_price_height = bbox_total[2] - bbox_total[0], bbox_total[3] - bbox_total[1]

                # 패딩 설정 (좌우, 상하)
                padding_total = 10

                # 박스 크기 계산
                box_total_width = total_price_width + 2 * padding_total
                box_total_height = total_price_height + 2 * padding_total

                # "총 가격" 박스의 왼쪽 위 좌표 계산 (오른쪽 하단에 10px 여백)
                box_total_x = im0.shape[1] - box_total_width - 10
                box_total_y = im0.shape[0] - box_total_height - 10

                # "총 가격" 박스 그리기 및 텍스트 중앙 배치
                draw_centered_text_box(draw, (box_total_x, box_total_y), total_price_text, font_bold, box_fill=(255, 255, 255), box_outline=(0, 0, 0), padding=padding_total)

                # "구매 항목" 텍스트 박스 위치 계산 및 표시
                # 시작 Y 좌표을 "총 가격" 박스 위로 설정, 간격 10px 추가
                item_y = box_total_y - 10

                # 각 구매 항목을 역순으로 표시하여 최신 항목이 위에 오도록 함
                for item in reversed(purchased_items):
                    # 각 항목의 텍스트 경계 상자 계산 (굵은 폰트 사용)
                    bbox_item = draw.textbbox((0, 0), item, font=font_bold)
                    item_width, item_height = bbox_item[2] - bbox_item[0], bbox_item[3] - bbox_item[1]

                    # 패딩 설정
                    padding_item = 10

                    # 박스 크기 계산
                    box_item_width = item_width + 2 * padding_item
                    box_item_height = item_height + 2 * padding_item

                    # "구매 항목" 박스의 왼쪽 위 좌표 계산 (오른쪽 끝에서 10px 여백)
                    box_item_x = im0.shape[1] - box_item_width - 10
                    box_item_y = item_y - box_item_height

                    # "구매 항목" 박스 그리기 및 텍스트 중앙 배치
                    draw_centered_text_box(draw, (box_item_x, box_item_y), item, font_bold, box_fill=(255, 255, 255), box_outline=(0, 0, 0), padding=padding_item)

                    # 다음 항목을 위에 표시하기 위해 Y 좌표 업데이트 (박스 높이 + 간격 5px)
                    item_y = box_item_y - 5

                # Pillow 이미지를 다시 OpenCV 이미지로 변환
                im0 = np.array(im_pil)

                # 창에 이미지 표시
                if view_img:
                    cv2.imshow(window_name, im0)
                    # `cv2.waitKey`로 키 입력을 감지 (딜레이를 10으로 설정)
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q'):
                        print("q 키가 눌렸습니다. 프로그램을 종료합니다.")
                        break  # 루프 종료

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        ... [Docstring 생략]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
