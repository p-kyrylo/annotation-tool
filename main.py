from ultralytics import YOLOWorld
from fastapi import FastAPI
import numpy as np
import argparse
import cv2
import sys
import os

app = FastAPI()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', help='path to the videofile', type=str)
    parser.add_argument('prompt', help='prompt, which specifies what objects to detect', type=str)
    parser.add_argument('save_path', help='specify path to save annotations', type=str)
    parser.add_argument('--nframe', help='specify which frames to leave (for example if 2 is specified, every 2nd frame will be annotated)', type=int)
    #parser.add_argument('--conf', help='specify confidence threshhold (default 0.01)', type=float)
    #parser.add_argument('--iou', help='specify intersection over union (default 0.1)', type=float)
    parser.add_argument('--only_txt', help='specify True if only .txt files are needed', type=bool)
    args = parser.parse_args()

    if not os.path.isfile(args.input_path):
        return None
    if args.nframe is not None and args.nframe <= 0:
        return None
    return (args.input_path, args.prompt, args.save_path, args.nframe, args.only_txt)

def is_empty(arr):
    if arr.size > 0:
        return True

def detection_percentage(results):
    detections = 0
    for result in results:
        if is_empty(result.boxes.cls.numpy()):
            detections = detections + 1
    
    return f'{np.around(detections/len(results)*100, 2)}%'

# get video file, parse it into frames
def to_frames(video):
    vid_capture = cv2.VideoCapture(video)
    if not vid_capture.isOpened():
            return None
    
    frames = []
    while vid_capture.isOpened():

        ret, frame = vid_capture.read()
        if not ret:
            break
        frames.append(frame)

    return frames

def skip_frames(frames):
    counter = 0
    index = 0
    if len(sys.argv) == 4:
        for frame in frames:
            index += 1 
            counter += 1
            if counter != int(sys.argv[3]):
                np.delete(frame, index)
                counter = 0

# annotate each frame using YOLO world
def annotate(frames, prompt, nframe):
    # Intializing model
    model = YOLOWorld('yolov8m-world.pt')
    # Setting prompt to specify which object to annotate 
    model.set_classes([prompt])

    counter = 1
    index = 0
    results = []
    for frame in frames:

        index += 1 
        counter += 1
        if nframe is not None:
            if counter == nframe:
                results += model.predict(frame, conf=0.01, iou=0.1)
                counter = 1
        else:
            results += model.predict(frame, conf=0.01, iou=0.1)
        
    return results

# save output in YOLO text format
def save(results, folder_name, only_txt=False):
    if os.path.exists(folder_name):
        for file in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file)
            os.remove(file_path)
    else:
        os.mkdir(folder_name)

    if not only_txt: 
        for i in range(len(results)):
            results[i].save(f'{folder_name}/frame_{i}.jpeg')
            results[i].save_txt(f'{folder_name}/frame_{i}.txt')
    else:
         for i in range(len(results)):
            results[i].save_txt(f'{folder_name}/frame_{i}.txt')

def main():
    params = parse_args()
    input_path, prompt, save_path, nframe, only_txt = params
    frames = to_frames(input_path)
    result = annotate(frames, prompt, nframe)
    save(result, save_path, only_txt)
    metrics = detection_percentage(result)
    print(metrics)

if __name__ == '__main__':
    main()
    

    
    
    