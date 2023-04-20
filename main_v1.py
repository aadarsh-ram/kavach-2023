import tempfile
import cv2
import gc
import numpy as np
import argparse
import easyocr
from ultralytics import YOLO
from tqdm.auto import tqdm

from src.keras_utils import load_model, detect_lp_width
from src.utils import im2single
from lp_filters import lp_filter

from Utilities.io import DataLoader
from Utilities.painter import Visualizer
from Models.RRDBNet import RRDBNet# we use RRDB in this demo
import tensorflow as tf

lp_threshold = 0.35
WPODResolution = 480
ocr_input_size = [80, 240] # Desired LP size (width x height)
all_imgs = []

MODEL_PATH = 'Pretrained/rrdb'
srgan_model = RRDBNet(blockNum=10)
srgan_model.load_weights(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)
model = YOLO('yolov8n.pt')
iwpod_net = load_model('iwpod_net/weights/iwpod_net')

def upscale(x):
    x = tf.convert_to_tensor(np.asarray([x/255.0]), dtype=tf.float32)
    x = srgan_model.predict(x)[0] * 255.0
    x = x.astype(np.uint8)
    return x

results = model.track(source="traffic.mp4", show=True, tracker="bytetrack.yaml") 

for result in tqdm(results):
    if result.boxes.id is None:
        image = result.orig_img
        all_imgs.append(image)
        continue

    boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
    confidences = result.boxes.conf.to('cpu').numpy().astype(float) 
    labels = result.boxes.cls.to('cpu').numpy().astype(int)
    id = result.boxes.id.to('cpu').numpy().astype(int)
    image = result.orig_img

    for ids, box, conf, label in zip(id, boxes, confidences, labels):
        if label not in [2, 4, 5, 7]: # result 2: 'car', 3: 'motorcycle', 5: 'bus' 7: 'truck'
            continue

        x_min, y_min, x_max, y_max = box
        image_crop = image[y_min:y_max, x_min:x_max]
        # print("Upscaling...")
        # image_crop = upscale(image_crop)
        # print("Finished Upscaling...")

        Ivehicle = image_crop.copy()

        iwh = np.array(Ivehicle.shape[1::-1],dtype=float).reshape((2,1))

        lp_output_resolution = tuple(ocr_input_size[::-1])

        # Runs IWPOD-NET. Returns list of LP data and cropped LP images
        
        # For dim = 0 error case
        if 0 in Ivehicle.shape:
            continue

        print(Ivehicle.shape, im2single(Ivehicle).shape)
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
                # image = cv2.putText(image, f'PLATE NUMBER: {plate1} CLASS: {label} ID: {ids}', ( x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
            else:
                print ("Plate:", plate2)
                # image = cv2.putText(image, f'PLATE NUMBER: {plate2} CLASS: {label} ID: {ids}', ( x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)

        # Bounding box
        cv2.putText(image, f'CLASS: {label} ID: {ids}', ( x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
        image = cv2.rectangle(image, ( x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3)

    all_imgs.append(image)

# Save video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (all_imgs[0].shape[1], all_imgs[0].shape[0]))
for img in all_imgs:
    out.write(img)
out.release()