import tempfile
import cv2
import numpy as np
import argparse
import easyocr
from src.keras_utils import load_model, detect_lp_width
from src.utils import im2single
from lp_filters import lp_filter
from ultralytics import YOLO

reader = easyocr.Reader(['en'], gpu=True)
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', type=str, help='Input Image')
parser.add_argument('-t', '--lp_threshold', type=float, default = 0.35, help = 'Detection Threshold')
args = parser.parse_args()

results = model.predict(args.image, classes=[2, 3, 5, 7])  # predict on an image
image = cv2.imread(args.image)

for result in results:
    boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
    confidences = result.boxes.conf.to('cpu').numpy().astype(float) 
    labels = result.boxes.cls.to('cpu').numpy().astype(int)
    cnt = 0
    for box, conf, label in zip(boxes, confidences, labels):
        x_min, y_min, x_max, y_max = box
        image_crop = image[y_min:y_max, x_min:x_max]
        lp_threshold = args.lp_threshold
        ocr_input_size = [80, 240] # desired LP size (width x height)
        
        # Loads network and weights
        iwpod_net = load_model('iwpod_net/weights/iwpod_net')

        Ivehicle = image_crop.copy()
        vtype = model.names[label]
        iwh = np.array(Ivehicle.shape[1::-1],dtype=float).reshape((2,1))

        if (vtype in ['car', 'bus', 'truck']):
            # Defines crops for car, bus, truck based on input aspect ratio (see paper)
            ASPECTRATIO = max(1, min(2.75, 1.0*Ivehicle.shape[1]/Ivehicle.shape[0]))  # width over height
            WPODResolution = 256 # faster execution
            lp_output_resolution = tuple(ocr_input_size[::-1])

        else:
            # Defines crop for motorbike  
            ASPECTRATIO = 1.0 # width over height
            WPODResolution = 208
            lp_output_resolution = (int(1.5*ocr_input_size[0]), ocr_input_size[0]) # for bikes, the LP aspect ratio is lower

        # Runs IWPOD-NET. Returns list of LP data and cropped LP images
        Llp, LlpImgs, _ = detect_lp_width(iwpod_net, im2single(Ivehicle), WPODResolution*ASPECTRATIO, 2**4, lp_output_resolution, lp_threshold)
        for i, img in enumerate(LlpImgs):
            orig_img = img * 255

            # Save cropped license plate into a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg') as temp:
                cv2.imwrite(temp.name, orig_img)
                orig_img = cv2.imread(temp.name)
            
            new_img = lp_filter(orig_img)
            result = reader.readtext(new_img)
            print (result)