import json
import numpy as np
import supervision as sv
from supervision import Position
import argparse
from collections import Counter
import cv2
from mapping import *

# This script provides utilities for processing video frames to detect and track people, update their attributes,
# and save the results in JSON format.

# Encoder class used to convert types incompatible with JSON serialization (e.g., NumPy data types)
# into standard types for saving to a .json file.
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert NumPy integers to Python int
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        # Convert NumPy floats to Python float
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        # Convert NumPy arrays to Python lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Default behavior for other data types
        return super().default(obj)

# Initializes a dictionary to represent a person detected in the video.
def init_person(id: int):
    print("New person detected with ID:", id)
    person = dict()

    # id
    person["id"] = id

    # gender
    person["gender"] = None
    person["gender_preds"] = list()

    # bag
    person["bag"] = None
    person["bag_prob"] = 0

    # hat
    person["hat"] = None
    person["hat_preds"] = list()

    # lines information
    person["line1_passages"] = 0
    person["line2_passages"] = 0

    # inference information
    person["first_inference"] = False
    person["last_frame"] = None
    person["overlap"] = False

    # Initialize lines_crossed
    person["trajectory"] = []

    person["previous_centers"] = [None, None]
    person["current_center"] = [None, None]

    return person

# Updates the attributes of a person based on model inference results.
def update_person_info(person: dict, infer_results):
    # Update gender prediction
    person["gender_preds"].append(infer_results["gender"].lower())
    gender_counter = Counter(person["gender_preds"])
    person["gender"] = gender_counter.most_common(1)[0][0]

    # Updating bag value 
    if person.get("bag") is None or person.get("bag") is False:
        person["bag"] = True if infer_results["bag"] == "yes" else False

    # Update hat attribute
    person["hat_preds"].append(True if infer_results["hat"] == "yes" else False)
    hat_counter = Counter(person["hat_preds"])
    person["hat"] = hat_counter.most_common(1)[0][0]

# Function to format JSON with aligned `trajectory`
def custom_dumps(data):
    """Custom JSON serialization that keeps `trajectory` lists aligned with other keys."""
    # Serialize the data to JSON with indentation
    json_str = json.dumps(data, cls=NpEncoder, indent=4)
    # Ensure `trajectory` lists are on a single line and aligned with other keys
    lines = json_str.splitlines()
    for i, line in enumerate(lines):
        if '"trajectory": [' in line:  # Detect the start of the trajectory list
            j = i
            while not lines[j].strip().endswith("]"):  # Find the end of the list
                j += 1
            # Combine all lines for this list into a single line
            combined = "".join(line.strip() for line in lines[i:j + 1])
            combined = combined.replace(", ", ",").replace("[ ", "[").replace(" ]", "]")
            # Align `trajectory` with the rest of the keys
            lines[i] = lines[i].split('"trajectory":')[0] + combined.strip()
            del lines[i + 1:j + 1]  # Remove the now redundant lines
    return "\n".join(lines)

# Function to save the detected people information to a JSON file
def save_output(output_filename: str, people_dict: dict):
    print(f"Saving results to {output_filename}...", end=" ")

    output_dict = {"people": []}

    # Keys to retain in the output
    output_keys = ["id", "gender", "hat", "bag", "trajectory"]

    # Filter each person's data to retain only specified keys
    for value in people_dict.values():
        filtered_value = {key: value[key] for key in output_keys if key in value}
        output_dict["people"].append(filtered_value)

    # Write the formatted JSON to the output file
    with open(output_filename, 'w') as fp:
        fp.write(custom_dumps(output_dict))

    print("done")

# Function that converts the results obtained from the detector to the format
# needed by the tracker object.
def get_dets(results):
    num_predictions = len(results[0])
    dets = np.zeros([num_predictions, 6], dtype=np.float32)
    for ind, object_prediction in enumerate(results[0].cpu()):
        dets[ind, :4] = np.array(object_prediction.boxes.xyxy, dtype=np.float32)
        dets[ind, 4] = object_prediction.boxes.conf
        dets[ind, 5] = object_prediction.boxes.cls

    return dets

# This function takes in input an image and applies on it a gamma correction with gamma
# passed as parameter, then returns the modified image.
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Parses command-line arguments to retrieve file paths for input video, configuration, and output.
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", dest="input_video_path")
    parser.add_argument("--configuration", dest="configuration_path")
    parser.add_argument("--results", dest="output_json_path")

    opt = parser.parse_args()
    return opt

def IoU(xyxyA, xyxyB):
    boxA = [xyxyA[0], xyxyA[1], xyxyA[2] - xyxyA[0], xyxyA[3] - xyxyA[1]]
    boxB = [xyxyB[0], xyxyB[1], xyxyB[2] - xyxyB[0], xyxyB[3] - xyxyB[1]]
    x = max(boxA[0], boxB[0])
    y = max(boxA[1], boxB[1])
    w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
    h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y

    # iou = area overlap / area of union
    iou = 0

    foundIntersect = True
    if w < 0 or h < 0:
        foundIntersect = False

    if foundIntersect:
        overlap_area = w*h
        union_area = boxA[2]*boxA[3] + boxB[2]*boxB[3] - overlap_area
        iou = overlap_area / union_area

    return iou

def get_coordinates(conf_path):
    # Load the configuration file
    with open(conf_path, "r") as f:
        line_dict = json.load(f)

    # Extract all coordinates
    x_real = []
    y_real = []

    for line_info in line_dict["lines"]:
        x_real.extend([float(line_info["x1"]), float(line_info["x2"])])
        y_real.extend([float(line_info["y1"]), float(line_info["y2"])])

    # Convert coordinates into NumPy arrays
    x_real = np.array(x_real)
    y_real = np.array(y_real)
    print(f"X_real: {x_real}")
    print(f"Y_real: {y_real}")

    # Pass the points to the mapping_lines function
    u, v = mapping_lines(x_real, y_real, conf_path)

    # Divide u and v into pairs (start and end for each line)
    lines_list = []
    for i, line_info in enumerate(line_dict["lines"]):
        id = int(line_info["id"])
        x1, x2 = int(u[2 * i]), int(u[2 * i + 1])
        y1, y2 = int(v[2 * i]), int(v[2 * i + 1])

        # Add the line to the list
        lines_list.append((id, (x1, y1), (x2, y2)))

    return lines_list

def line_equation(point1, point2):
    """
    Calculate the slope and intercept of the line passing through two points.

    Parameters:
        point1 (list): First point [x1, y1].
        point2 (list): Second point [x2, y2].

    Returns:
        tuple: (slope, intercept) where:
               - slope is the line's slope.
               - intercept is the y-intercept (or None if the line is vertical).
    """
    if point2[0] - point1[0] == 0:  # Vertical line case
        return None, point1[0]
    
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])  # Calculate slope
    q = point1[1] - m * point1[0]  # Calculate intercept

    return m, q

def check_crossing(curr, prev, line):
    """
    Check if a trajectory (current and previous positions) crosses a line.

    Parameters:
        curr (list): Current position [x, y].
        prev (list): Previous position [x, y].
        line (tuple): Line information (id, (x1, y1), (x2, y2)).

    Returns:
        bool: True if the trajectory crosses the line, otherwise False.
    """
    id, (x1, y1), (x2, y2) = line
    if prev == [None, None]:  # If no previous position is available
        return False

    # Coordinates of the person's trajectory (bounding box)
    traj_x1, traj_y1 = prev
    traj_x2, traj_y2 = curr

    # Calculate the slope and intercept for the line and the trajectory
    vl_slope, vl_intercept = line_equation([x1, y1], [x2, y2])
    traj_slope, traj_intercept = line_equation(prev, curr)

    # Check if the lines are parallel
    if vl_slope == traj_slope:
        return False

    # Calculate the intersection point
    if vl_slope is None:  # Reference line is vertical
        x = vl_intercept
        y = traj_slope * x + traj_intercept
    elif traj_slope is None:  # Trajectory is vertical
        x = traj_intercept
        y = vl_slope * x + vl_intercept
    else:  # General case
        x = (traj_intercept - vl_intercept) / (vl_slope - traj_slope)
        y = vl_slope * x + vl_intercept
        
    # Verify that the intersection is within the limits of the line and trajectory
    return (
        (min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)) and
        (min(traj_x1, traj_x2) <= x <= max(traj_x1, traj_x2) and min(traj_y1, traj_y2) <= y <= max(traj_y1, traj_y2)) and
        ((x2 - x1) * (traj_y2 - traj_y1) - (y2 - y1) * (traj_x2 - traj_x1) < 0)
    )




    

