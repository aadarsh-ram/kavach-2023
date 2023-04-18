import tempfile
import cv2
import numpy as np
import argparse
import easyocr
from ultralytics import YOLO

# Custom imports
from src.keras_utils import load_model, detect_lp_width
from src.utils import im2single
from lp_filters import lp_filter

reader = easyocr.Reader(['en'], gpu=True)
model = YOLO("yolov8m.pt")  # Load pretrained YoloV8m model

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', type=str, help='Input Image')
parser.add_argument('-t', '--lp_threshold', type=float, default = 0.35, help = 'Detection Threshold')
args = parser.parse_args()

results = model.predict(args.image, classes=[2, 3, 5, 7])  # Predict only vehicle classes
image = cv2.imread(args.image)

for result in results:
    boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
    confidences = result.boxes.conf.to('cpu').numpy().astype(float) 
    labels = result.boxes.cls.to('cpu').numpy().astype(int)

    for box, conf, label in zip(boxes, confidences, labels):
        # Crop vehicle
        x_min, y_min, x_max, y_max = box
        image_crop = image[y_min:y_max, x_min:x_max]
        lp_threshold = args.lp_threshold
        ocr_input_size = [80, 240] # Desired LP size (width x height)
        
        # Loads network and weights
        iwpod_net = load_model('iwpod_net/weights/iwpod_net')

        Ivehicle = image_crop.copy()
        # vtype = model.names[label] # Vehicle type
        iwh = np.array(Ivehicle.shape[1::-1],dtype=float).reshape((2,1))

        WPODResolution = 480
        lp_output_resolution = tuple(ocr_input_size[::-1])

        # Runs IWPOD-NET. Returns list of LP data and cropped LP images
        Llp, LlpImgs, _ = detect_lp_width(iwpod_net, im2single(Ivehicle), WPODResolution, 2**4, lp_output_resolution, lp_threshold)
        for i, img in enumerate(LlpImgs):
            orig_img = img * 255

            # Save cropped license plate into a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg') as temp:
                cv2.imwrite(temp.name, orig_img)
                orig_img = cv2.imread(temp.name)

            # Apply license plate filters
            new_img1, new_img2 = lp_filter(orig_img)
            try:
                boxes1, plate1, conf1 = reader.readtext(new_img1)[-1]
            except:
                boxes1, plate1, conf1 = None, '', 0
            try:
                boxes2, plate2, conf2 = reader.readtext(new_img2)[-1]
            except:
                boxes2, plate2, conf2 = None, '', 0

            # Plate with higher confidence is returned
            if (conf1 > conf2):
                print ("Plate:", plate1)
            else:
                print ("Plate:", plate2)