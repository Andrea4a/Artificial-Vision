from pathlib import Path

import torch.cuda
from ultralytics import YOLO
from boxmot import BoTSORT

from inference import inference
from utils.utils import *
import math

# Parse command-line arguments
args = parse_opt()

# Dictionary to store information about detected people
people_det = dict()

# ---------- FILE PATHS ----------
# Define file paths based on user input
OUTPUT_FILENAME = args.output_json_path
VIDEO_PATH = args.input_video_path
CONFIGURATION_FILE_PATH = args.configuration_path

# ---------- VIDEO INFO ----------
# Skipping frames to reduce computational load
FRAME_TO_SKIP = 4
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
FRAME_RATE = int(video_info.fps / FRAME_TO_SKIP)
INFERENCE_RATE = FRAME_RATE * 2  # one inference every 2 seconds
WIDTH_RESOLUTION = video_info.width
HEIGHT_RESOLUTION = video_info.height

# Print video details
print(f"Video information:\n"
      f"\t- width: {WIDTH_RESOLUTION}\n"
      f"\t- height: {HEIGHT_RESOLUTION}\n"
      f"\t- fps: {FRAME_RATE}\n")

# ---------- LINES INFO AND CONFIGURATION ----------

# Initialize lines from the configuration file and reset passage counters
lines_list = get_coordinates(CONFIGURATION_FILE_PATH)
line_passages = {}

# ---------- COLORS AND CONFIGURATION ----------

# Define colors and configurations for drawing
LINE_COLOR = (0, 0, 0)  # black
LINES_COLOR = (190, 115, 0)  # blue
BBOX_COLOR = (0, 0, 255)  # red
WHITE = (255, 255, 255)  # white

# information for drawing part
THICKNESS = 3
FONTSCALE = 0.6
DX, DY = 5, 5  # delta x and delta y values used for drawing

# Print initialized lines
print(f"Lines list: ", lines_list)

print(f"Lines information:")
for i, line in enumerate(lines_list):
    print(f"\t{i+1}:\n"
          f"\t\t- (x_start, y_start): {line[1]}\n"
          f"\t\t- (x_end, y_end): {line[2]}")

# ---------- TRACKER ----------

# Configure the tracking system
DEVICE_STRING = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STRING)
FP16 = True if (DEVICE_STRING != "cpu") else False
TRACK_BUFFER = FRAME_RATE * 5  # 5 seconds

# Initialize the tracker with specified configurations
tracker = BoTSORT(
    model_weights=Path('models/osnet_x0_25_msmt17.pt'),
    device=DEVICE,
    fp16=FP16,
    with_reid=True,
    track_buffer=TRACK_BUFFER,
    new_track_thresh=0.8,
    frame_rate=FRAME_RATE
)

print(f"Tracker information:\n"
      f"\t- type: BotSORT with reID\n"
      f"\t- device: {DEVICE_STRING}\n"
      f"\t- FP16: {FP16}\n")

# ---------- DETECTOR ----------

# Initialize YOLO model for detection
model = YOLO("models/yolov8m.pt", verbose=False)

# Open the input video file
vid = cv2.VideoCapture(VIDEO_PATH)
current_frame = 0

# Process video frame by frame
while True:
    ret, img = vid.read()
    current_frame += FRAME_TO_SKIP
    vid.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    if ret:
        frame = img.copy() # Create a copy of the current frame

     # Perform detection using YOLO
        results = model(img, classes=0, device=DEVICE, verbose=False)
        dets = get_dets(results) # Convert detection results to tracker format


        # Update tracker with the new detections
        tracks = tracker.update(dets, img)  # --> (x, y, x, y, id, conf, cls, ind)

        # drawing lines on the frame
        for i, line in enumerate(lines_list):
            line_text = str(line[0])
            if line[0] not in line_passages:
                line_passages[line[0]] = 0
            

            # Determines the lower right point between the two ends of the line
            if (line[1] > line[2]) or (line[1] == line[2] and line[1] > line[2]):
                reference_point = (
                    min(line[1][0], line[2][0]),  # Minimo della coordinata X
                    min(line[1][1], line[2][1])   # Minimo della coordinata Y
                )# First point is further down on the right
            else:
                reference_point = (
                    min(line[1][0], line[2][0]),  # Minimo della coordinata X
                    max(line[1][1], line[2][1])   # Minimo della coordinata Y
                )
            
            text_position = (reference_point[0] - 30, reference_point[1] - 30)
            text_position = (
                max(text_position[0], 0),  
                max(text_position[1], 0)   
            )  # Second point is further down on the right

            # Calcola la larghezza e l'altezza del testo
            (text_width, text_height), baseline = cv2.getTextSize(
                line_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,  # Scala del testo
                THICKNESS
            )

            x, y = text_position
            rect_top_left = (x, y - text_height - baseline)  # Angolo superiore sinistro
            rect_bottom_right = (x + text_width, y + baseline)  # Angolo inferiore destro

            # Assicurati che il rettangolo rimanga visibile
            rect_top_left = (max(rect_top_left[0], 0), max(rect_top_left[1], 0))
            rect_bottom_right = (
                min(rect_bottom_right[0], img.shape[1] - 1),
                min(rect_bottom_right[1], img.shape[0] - 1)
            )
            # Add line label and draw the line

            img = cv2.rectangle(
                img,
                rect_top_left,
                rect_bottom_right,
                WHITE,
                -1  # filling
            )

            # Add line label and draw the line
            cv2.putText(
                img,
                line_text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                LINES_COLOR,
                THICKNESS
            )

            img = cv2.line(
                img,
                line[1],
                line[2],
                LINES_COLOR,
                THICKNESS
            )
            # Calculate the midpoint of the line
            center_point = (
                (line[1][0] + line[2][0]) // 2,
                (line[1][1] + line[2][1]) // 2
            )

            # Calculate the direction of the arrow
            direction = (
                line[2][0] - line[1][0],
                line[2][1] - line[1][1]
            )

            if direction[0]:
                arrow_direction = (direction[1], -direction[0])
            else:
                arrow_direction = (-direction[1], direction[0])

            # Normalize the direction
            norm = (arrow_direction[0]**2 + arrow_direction[1]**2)**0.5
            unit_direction = (arrow_direction[0] / norm, arrow_direction[1] / norm)

            # Calculate the end point of the arrow starting from the center
            arrow_length = 100
            arrow_end = (
                int(center_point[0] + arrow_length * unit_direction[0]),
                int(center_point[1] + arrow_length * unit_direction[1])
            )

            # Disegna la freccia
            img = cv2.arrowedLine(
                img,
                center_point,
                arrow_end,
                LINES_COLOR,
                THICKNESS,
                tipLength=0.3
            )

        # Process tracked objects if any are detected
        if tracks.shape[0] != 0:
            xyxys = tracks[:, 0:4].astype('int')  # float64 to int
            ids = tracks[:, 4].astype('int')  # float64 to int
            confs = tracks[:, 5]
            clss = tracks[:, 6].astype('int')  # float64 to int
            inds = tracks[:, 7].astype('int')  # float64 to int

            for i, (xyxy, id, conf, cls) in enumerate(zip(xyxys, ids, confs, clss)):
                person = people_det.get(id)
                if person is None:
                    # Initialize new person if not already tracked
                    person = init_person(id)
                    people_det[id] = person
                
                # Extract bounding box coordinates
                X, Y, X2, Y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                cropped_image = person["last_frame"] = frame[Y:Y2, X:X2]

                for j in range(i + 1, len(ids)):
                    xyxyA = xyxy
                    xyxyB = xyxys[j]
                    iou = IoU(xyxyA, xyxyB)
                    personA = person
                    personB = people_det.get(ids[j])
                    if personB is None:
                        personB = init_person(ids[j])
                        people_det[ids[j]] = personB
                    if iou < 0.3:
                        personA["overlap"] = False
                        personB["overlap"] = False
                    else:
                        personA["overlap"] = True
                        personB["overlap"] = True
                        break

                # Calculates the lower center of the current bounding box
                current_center = (
                    (xyxy[0] + xyxy[2]) // 2,  # Coordinate X del centro
                    xyxy[3]                    # Coordinate Y del centro inferiore
                )
                person["current_center"] = current_center

                # Perform inference on cropped image if valid
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cropped_image = adjust_gamma(cropped_image, gamma=1.5)
                    infer_results = inference(cropped_image)
                    update_person_info(person=person, infer_results=infer_results)
                    person["first_inference"] = True

                for i, line in enumerate(lines_list):
                    if check_crossing(person["current_center"], person["previous_centers"], line):
                        person[f"line{line[0]}_passages"] += 1
                        line_passages[line[0]] += 1  # Aggiorna il contatore nel dizionario
                        person["trajectory"].append(line[0])  # Aggiungi la linea attraversata alla traiettoria

                # Update the bottom center for the next frame
                person["previous_centers"] = person["current_center"]

                # getting labels for information drawing
                id_label = str(person["id"])
                gender_label = person['gender'] if person['gender'] is not None else "?"
                label = (
                        "?" if person['bag'] is None and person['hat'] is None
                        else "Bag" if person['bag'] and person['hat'] is None
                        else "Hat" if person['hat'] and person['bag'] is None
                        else "Bag Hat" if person['bag'] and person['hat']
                        else "Bag" if person['bag']
                        else "Hat" if person['hat']
                        else "No Bag No Hat"
                )

                # top left person bounding box, id
                x_id, y_id = xyxy[0] + DX, xyxy[1] + DY
                x2_id, y2_id = x_id + 30, y_id + 30
                img = cv2.rectangle(
                    img,
                    (x_id, y_id),
                    (x2_id, y2_id),
                    WHITE,
                    -1  # filling
                )

                x_pos = int((x_id+x2_id)/2 - DX) if person["id"] < 10 else int((x_id+x2_id)/2 - 2*DX)
                img = cv2.putText(
                    img,
                    id_label,
                    (x_pos, int((y_id+y2_id)/2) + DY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONTSCALE,
                    BBOX_COLOR,
                    THICKNESS
                )

                # Check if "lines_crossed" exists and create a string to add
                trajectory_text = ""
                if "trajectory" in person and person["trajectory"] :
                    trajectory_text = f"[{', '.join(map(str, person['trajectory']))}]"

                # bottom person bounding box, all info
                text = (f"Gender: {gender_label[0].upper()}\n"
                        f"{label}\n"
                        f"{trajectory_text}")

                # drawing person bounding box
                img = cv2.rectangle(
                    img,
                    (xyxy[0], xyxy[1]),
                    (xyxy[2], xyxy[3]),
                    BBOX_COLOR,
                    THICKNESS
                )
                 # Draw a rectangle below the bounding box to display additional information
                x_id, y_id = xyxy[0] + DX, xyxy[3] + DY
                x2_id, y2_id = x_id + 180, y_id + 60
                img = cv2.rectangle(
                    img,
                    (x_id, y_id), # Top-left corner of the rectangle
                    (x2_id, y2_id),  # Bottom-right corner of the rectangle
                    WHITE, # Color of the rectangle (filled with white)
                    -1  # filling
                )

                # drawing person information near its bounding box
                x, y0, dy = xyxy[0] + 2*DX, xyxy[3] + 2*DY + 8, 20
                for i, line in enumerate(text.split('\n')):
                    y = y0 + i * dy
                    
                    # Add text information inside the rectangle
                    cv2.putText(
                        img,
                        line,
                        (x, y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        LINE_COLOR,
                        1
                    )

        # drawing general information on top left screen
        img = cv2.rectangle(
            img,
            (0, 0),
            (400, 165),
            WHITE,
            -1  # filling
        )
        # Display general information such as total people and line crossings
        total_persons = tracks.shape[0] # Total number of currently tracked people
        text = f"Total people: {total_persons}\n"
        for line_id, passages in line_passages.items():
            text += f"Passages for line {line_id}: {passages}\n"
        
        
        # Define position for displaying general information
        x, y0, dy = 20, 40, 30  # Initial coordinates and line spacing
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy  # Calculate vertical position for each line of text

            cv2.putText(
                img,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                LINE_COLOR,
                2
            )

        # show image with bboxes, ids, classes and confidences
        cv2.imshow('frame', img)

        #frame refreshing
        cv2.waitKey(1)
    else:
        break

# ----------- FINALIZATION ----------
vid.release()
cv2.destroyAllWindows()

# ----------- OUTPUT SAVING ----------
#  saving people information in json file
save_output(OUTPUT_FILENAME, people_det)
