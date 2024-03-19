from ultralytics import YOLOWorld
import cv2
import sys
import os

# get video file, parse it into frames

def get_fpath():
    if len(sys.argv) == 2:
        return sys.argv[1]
    
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

# annotate each frame using YOLO world
def annotate(frames):
    # Intializing model
    model = YOLOWorld('yolov8m-world.pt')
    # Setting prompt to specify which object to annotate 
    model.set_classes(['steel wet cans'])

    results = []
    for frame in frames:
         results += model.predict(frame)

    return results


# save output in YOLO text format

def save(results, folder_name):
    if os.path.exists(folder_name):
        for file in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file)
            os.remove(file_path)
    else:
        os.mkdir(folder_name)
    for i in range(len(results)):
        results[i].save(f'{folder_name}/frame_{i}.jpeg')
        formatted = f'{results[i].boxes.cls.numpy()} {results[i].boxes.xywhn.numpy()}'
        with open(f'{folder_name}/frame_{i}.txt', 'w') as file:
            file.write(formatted)
        

        
      
     
# validate results using 'verifyannotations' package
def main():
    video = get_fpath()

    frames = to_frames(video)

    result = annotate(frames)

    save(result, 'output')

if __name__ == '__main__':
    main()
    

    
    
    