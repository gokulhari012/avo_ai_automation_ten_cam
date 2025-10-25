import cv2
from ultralytics import YOLO, SAM
import logging
import numpy as np
from logger import log_info, log_error
import json 
import torch
import copy
import threading
import yolo_functions.dimension as dimension

# print("Torch CUDA Available:", torch.cuda.is_available())
# print("Torch Device Count:", torch.cuda.device_count())
# print("Torch Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA")
# print("OpenCV CUDA Enabled:", cv2.cuda.getCudaEnabledDeviceCount())

brush_dimensions_path = "./json/brush_dimensions.json"
settings_file_path = "./json/settings-old.json"

with open(brush_dimensions_path, "r") as file:
    brush_dimensions_json = json.load(file)

with open("configuration.json", "r") as file:
    json_data = json.load(file)
contours_area_threshold, contours_area_tolarance_percentage, width_height_ratio_tolarance_percentage = (0,0,0) # dummy values
# contours_area_threshold = json_data["contours_area_threshold"]
# contours_area_tolarance_percentage = json_data["contours_area_tolarance_percentage"]
# width_height_ratio_tolarance_percentage = json_data["width_height_ratio_tolarance_percentage"]

top_camera = json_data["top_camera"]
side_camera = json_data["side_camera"]
square_value = json_data["camera_limit_square_value"]
dimension_tolarance = json_data["dimension_tolarance"]

brush_id_name = ""
# Read the folder name from selected_brush.txt
with open("json/selected_brush.txt", "r") as file:
    brush_id_name = file.read().strip()

downscale=1
yolodownscale=2
# Set font and size
font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 2  # Increase font size for better visibility
font_thickness = 2  # Adjust thickness for better highlighting

logging.getLogger("ultralytics").setLevel(logging.ERROR) 
confidence = 0.40
defect_confidence = 0.50
square_value_detect = 150
square_skip = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)
print(torch.cuda.device_count())  # Get number of available GPUs
print(torch.cuda.get_device_name(0))

# Load the YOLOv8 model
common_knowledge_path = "common_knowledge/weights/best.pt"
# model_path = "runs/detect/train3/weights/last.pt"
common_knowledge_model = YOLO(common_knowledge_path).to(device)
# Get class names and colors
common_knowledge_class_names = common_knowledge_model.names

brush_knowledge_path = "brush_knowledge/weights/best.pt"
brush_knowledge_model = YOLO(brush_knowledge_path).to(device)
brush_knowledge_class_names = brush_knowledge_model.names

sam_model = SAM("sam/sam2_t.pt")
sam_model_lock = threading.Lock()

colors = {"cb":(0,255,0),"defect":(0,0,255)}

# Reference object for calibration (example)
real_world_size = 10  # in mm (for example, the reference object's real-world size)
pixel_size = 100  # in pixels (size of the reference object in the image)
conversion_factor = real_world_size / pixel_size



def load_variable():
    try:
        data = {}
        with open(settings_file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading setting json : {e}")
    

def load_values():
    global l_h1, l_s1, l_v1, h_h1, h_s1, h_v1, l_h2, l_s2, l_v2, h_h2, h_s2, h_v2
    data = load_variable()
    if data:
        l_h1, l_s1, l_v1, h_h1, h_s1, h_v1 = (data["l_h1"], data["l_s1"], data["l_v1"], data["h_h1"], data["h_s1"], data["h_v1"])
        l_h2, l_s2, l_v2, h_h2, h_s2, h_v2 = (data["l_h2"], data["l_s2"], data["l_v2"], data["h_h2"], data["h_s2"], data["h_v2"])

def is_top_camera(camera_index):
    if camera_index == top_camera:
        return True
    else:
        return False
    
def is_side_camera(camera_index):
    if camera_index == side_camera:
        return True
    else:
        return False

#The is very old one not used
def capture_cb(frame, brush_id):
    status = False
    # Perform detection
    # frame_temp = torch.tensor(frame, dtype=torch.float32).to(device)
    # # frame = frame.to(device)
    # results = common_knowledge_model(frame_temp)
    results = common_knowledge_model(frame)
    detections = results[0]
    
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    captured_frame = None
    captured_position = None
    captured_frame = np.copy(frame)
    
    # Draw bounding boxes
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = common_knowledge_class_names[class_id]

        #if True:
        if conf > confidence and class_name == 'cb':  # Modify condition as needed
            # Object center
            obj_center_x, obj_center_y = x1+((x2-x1)//2), y1+((y2-y1)//2)
            # Check if object is near center
            if abs(center_x-square_value) < obj_center_x and (center_x+square_value) > obj_center_x and abs(center_y-square_value) < obj_center_y and abs(center_y+square_value) > obj_center_y:
                print(f"Object '{class_name}' centered! Capturing image.")
                captured_position = x1, y1, x2, y2
                # cv2.imshow("Captured Image", frame)
                # cv2.waitKey(0)  # Show captured image for 500ms
                status = True

            color = colors[class_name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if class_name == 'cb': 
                label = f"Brush ID: {brush_id}: {conf:.2f}"
            else:
                label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    
    return frame, status, captured_frame, captured_position

# The below function it not used (countour detection)
def check_dimensions(frame, brush_id_name):
    cb_dimensions_good = True
    brush_id_name = brush_id_name+"_area"
    if brush_id_name in brush_dimensions_json:
        brush_dimensions = brush_dimensions_json[brush_id_name]
        # Capture frame from webcam
        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Get trackbar positions

        # Define color ranges
        lower1 = np.array([l_h1, l_s1, l_v1])
        upper1 = np.array([h_h1, h_s1, h_v1])
        lower2 = np.array([l_h2, l_s2, l_v2])
        upper2 = np.array([h_h2, h_s2, h_v2])

        # Create masks
        mask1 = cv2.inRange(image_hsv, lower1, upper1)
        mask2 = cv2.inRange(image_hsv, lower2, upper2)

        # Combine masks
        combined_mask = cv2.bitwise_or(mask1, mask2)

        # === SMOOTHING THE MASK ===
        kernel = np.ones((5, 5), np.uint8)  # Kernel for morphological operations

        # 1. Apply Gaussian Blur (smooths edges)
        blurred_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

        # 2. Apply Morphological Closing (fills small holes)
        closed_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Apply Morphological Opening (removes small noise)
        smoothed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours on the smoothed mask
        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_area = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > contours_area_threshold:  # Ignore small objects
                contours_area = cv2.contourArea(cnt)
                # Get minimum area rectangle
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                # Calculate width & height
                width = np.linalg.norm(box[0] - box[1])
                height = np.linalg.norm(box[1] - box[2])

                # Draw bounding box
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                # Display width & height
                cv2.putText(frame, f"W: {width:.1f}px", (box[0][0], box[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                cv2.putText(frame, f"H: {height:.1f}px", (box[1][0], box[1][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

                # Approximate contour with a polygon
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Draw polygon
                cv2.polylines(frame, [approx], True, (0, 0, 255), 2)

                # Measure point-to-point distances
                for j in range(len(approx)):
                    pt1 = tuple(approx[j][0])
                    pt2 = tuple(approx[(j + 1) % len(approx)][0])  # Next point (loop back)
                    distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
                    midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

                    # Display distance
                    cv2.putText(frame, f"{distance:.1f}px", midpoint, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)

        
        contours_area_tolarance = brush_dimensions*contours_area_tolarance_percentage/100
            # print("Mask Area: "+str(contours_area))
        if contours_area > contours_area_threshold:
            cb_dimensions_good = False
            if brush_dimensions - contours_area_tolarance <= contours_area <= brush_dimensions + contours_area_tolarance:
                cb_dimensions_good = True
        # print("Expected Dimension: "+str(brush_dimensions))
        # print("Contour Dimension: "+str(contours_area))
        log_info("Expected Dimension: "+str(brush_dimensions))
        log_info("Contour Dimension: "+str(contours_area))
        log_info("Tolarance: "+str(contours_area_tolarance)+" Diff: "+str(contours_area-brush_dimensions))

    return frame, cb_dimensions_good

load_values()

def check_frame(camera_index, frame, brush_count_id, brush_id, is_first_camera = False, is_training_call=False):
    cb_identified = False 
    defect_identified = False 
    cb_dimensions = (0,0)
    cb_dimensions_good = True
    dimension_area=0
    width_height_ratio=0
    dimensions = {}
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    # print("\n\n\n\n\n\n\n")
    # print(type(frame))
    # Perform detection
    # frame_temp = torch.tensor(frame, dtype=torch.float32).to(device)
    # # frame = frame.to(device)
    # results = brush_knowledge_model(frame_temp)
    results = brush_knowledge_model(frame)
    detections = results[0]
    label_confidence_map = {}
    label_value_map = {}
    defect_confidence_list =[]
    defect_value_list = []
    # Draw bounding boxes
    #log_info("Analys start")
    for box in detections.boxes:
        # print(box)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = brush_knowledge_class_names[class_id]
        #if True:
        # Modify condition as needed
        obj_center_x, obj_center_y = x1+((x2-x1)//2), y1+((y2-y1)//2)
        # Check if object is near center

        if square_skip or (abs(center_x-square_value_detect) <= obj_center_x <= abs(center_x+square_value_detect) and abs(center_y-square_value_detect) <= obj_center_y <= abs(center_y+square_value_detect)):
            if class_name == "brushID_0" or class_name == 'defect': # brushID_0 is defect
                if (not is_training_call) and (conf > defect_confidence):
                    label = f"defect: {conf:.2f}"
                    defect_confidence_list.append(label)
                    defect_value_list.append((x1, y1, x2, y2))
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)             
                    if not defect_identified:
                        defect_identified = True
                    log_info("Decfect identified 1")
            else :
                if conf > confidence:
                    label_confidence_map[class_name] = conf
                    label_value_map[class_name] = (x1, y1, x2, y2)
                    log_info("cp identified 1")


    if label_confidence_map and label_value_map:
        maxmium_confidence_brush = max(label_confidence_map, key=label_confidence_map.get)
        if not cb_identified and (maxmium_confidence_brush == brush_id or maxmium_confidence_brush == "cb"):
            cb_identified = True
            log_info("cp identified 3")

        log_info(maxmium_confidence_brush)
        color = colors["cb"]
        conf = label_confidence_map[maxmium_confidence_brush]
        x1, y1, x2, y2 =  label_value_map[maxmium_confidence_brush]
        width = x2 - x1
        height = y2 - y1
        # Convert dimensions from pixels to real-world units
        real_width = width * conversion_factor
        real_height = height * conversion_factor
        cb_dimensions = (real_width, real_height)
        log_info("Cb Dimention wXh: "+ str(cb_dimensions))

        if cb_identified:
            #frame_dimension, cb_dimensions_good, dimension_area, width_height_ratio = frame_process_check_dimension(camera_index, frame, brush_id_name, (x1, y1, x2, y2), is_training_call)
            frame_dimension, cb_dimensions_good, dimensions = frame_process_check_contour_dimension(camera_index, frame, brush_id_name, (x1, y1, x2, y2), is_training_call)

        obj_center_x, obj_center_y = x1+((x2-x1)//2), y1+((y2-y1)//2)
        cv2.circle(frame, (obj_center_x, obj_center_y), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"BrushID: {maxmium_confidence_brush} - CountID: {brush_count_id} - Conf: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        log_info("cp identified 2: "+str(cb_identified)+" Class Name: "+str(maxmium_confidence_brush))

    if defect_confidence_list and defect_value_list:
        color = colors["defect"]
        for i in range (0,len(defect_confidence_list)):
            label = defect_confidence_list[i]
            (x1, y1, x2, y2) = defect_value_list[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness) 

    # log_info("Analys End")
    
    if cb_identified and defect_identified:
        log_info("##############################Both identified####################################")
    #print(cb_identified, defect_identified)
    if is_training_call:
        return frame, cb_identified, dimension_area, width_height_ratio
    else:
        return frame, cb_identified, defect_identified, cb_dimensions, cb_dimensions_good, dimensions



def detect_objects(image):
    """Run YOLO object detection on the input image."""
    # image = image.to(device)  n
    # image_temp = torch.tensor(image, dtype=torch.float32).to(device)
    # yolo_results = brush_knowledge_model(image_temp)
    yolo_results = brush_knowledge_model(image)
    if yolo_results and len(yolo_results[0].boxes) > 0:
        return yolo_results[0].boxes.xyxy.cpu().numpy()
    return None

def filter_small_boxes(boxes, min_area=100):  # Adjust `min_area` as needed
    filtered_boxes = [box for box in boxes if (box[2] - box[0]) * (box[3] - box[1]) >= min_area]
    return np.array(filtered_boxes)  # Return filtered boxes as NumPy array

def segment_objects(image, boxes):
    """Run SAM segmentation on detected objects using bounding boxes."""
    with sam_model_lock:
        sam_results = sam_model(image, bboxes=boxes, verbose=False, save=True)
    if hasattr(sam_results[0], "masks") and sam_results[0].masks is not None:
        return sam_results[0].masks.data.cpu().numpy()
    return None

def process_mask(mask, frame_shape):
    """Smooth and preprocess the segmentation mask for better edge detection."""
    mask = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
    smooth_img = cv2.bilateralFilter(mask_resized, 5, 50, 50)
    blurred = cv2.GaussianBlur(smooth_img, (11, 11), 1.5)
    return blurred

def compute_pca_orientation(contour):
    """Compute PCA to determine the orientation angle of the detected object."""
    data_pts = np.array(contour, dtype=np.float32).reshape(-1, 2)
    mean, eigenvectors = cv2.PCACompute(data_pts, mean=None)
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * (180 / np.pi)
    return angle

def detect_and_measure_edges(mask, frame):
    """Detect object edges, compute distances, and determine orientation."""
    thresh1 = 50
    thresh2 = 100
    thresh2 = max(thresh2, 10)  # Ensure a minimum threshold value

    _, thresh = cv2.threshold(mask, thresh1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    line_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    dimensions = []
    width_height_ratio = 0
    for contour in contours:
        if cv2.contourArea(contour) > contours_area_threshold:  # Ignore small noise
            cropped, box, width, height = get_oriented_crop(frame, contour)
            dimensions.append(f"Width: {width}px, Height: {height}px")

            for i in range(len(box)):
                pt1 = tuple(box[i])
                pt2 = tuple(box[(i+1) % len(box)])
                cv2.line(frame, pt1, pt2, line_colors[i % len(line_colors)], 3)

            for i, point in enumerate(box):
                cv2.circle(frame, tuple(point), 5, (0, 0, 0), -1)

            epsilon = (1/thresh2) * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [tuple(point[0]) for point in approx]

            for i in range(len(points)):
                p1, p2 = points[i], points[(i + 1) % len(points)]
                distance = cv2.norm(np.array(p1) - np.array(p2))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f"{distance:.2f}px", ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)

            angle = compute_pca_orientation(contour)
            center = np.mean(box, axis=0).astype(int)
            length = 100
            angle_rad = np.deg2rad(angle)
            x2 = int(center[0] + length * np.cos(angle_rad))
            y2 = int(center[1] + length * np.sin(angle_rad))
            cv2.line(frame, tuple(center), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Angle: {angle:.2f}°", (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)

            y_offset = 20
            for i, dim in enumerate(dimensions):
                cv2.putText(frame, dim, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                y_offset += 20
            if width>height:
                width,height=height,width
            cropped=cv2.resize(cropped, (int(width*downscale),int(height*downscale)))
            width_height_ratio = height/width
            #cv2.resizeWindow("Oriented Cropped Object", width, height) 
            #cv2.imshow("Oriented Cropped Object"+str(cam_id), cropped)
    return width_height_ratio

def get_oriented_crop(image, contour):
    """Extract and crop the oriented bounding box from the image and ensure it's vertically aligned."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    # Ensure the biggest dimension is always the height (oriented vertically)
    

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (width, height))
    if width > height:
        cropped= cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    return cropped, box, width, height

def is_inside(box_outer, box_inner):
    x1, y1, x2, y2 = box_outer  # Outer box
    x1_p, y1_p, x2_p, y2_p = box_inner  # Inner box

    return (x1 <= x1_p) and (y1 <= y1_p) and (x2 >= x2_p) and (y2 >= y2_p)

def frame_process_check_dimension(camera_index, frame, brush_id_name, label_value_box, is_training_call):
    cb_dimensions_good = True
    contours_area = 0
    width_height_ratio = 0
    width_height_ratio_tolarance = 0
    contours_area_tolarance = 0
    brush_dimensions = 0
    brush_ratio = 0
    dimension_error = True
    ratio_error = True
    if not is_training_call:
            brush_id_name_area = brush_id_name+"_area"
            brush_dimensions = brush_dimensions_json[brush_id_name_area]

    if is_training_call or brush_id_name_area in brush_dimensions_json:
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        image_rgb=cv2.resize(image_rgb, (int(width/yolodownscale),int(height/yolodownscale)))
        boxes = detect_objects(image_rgb)
        # print(boxes)
        # Example usage:
        masks_temp = None
        if boxes is not None:
            filtered_boxes = filter_small_boxes(boxes, min_area=contours_area_threshold) 
            if filtered_boxes is not None and filtered_boxes.size != 0:
                # Ensure correct shape: (N, 4)
                # filtered_boxes = np.array(filtered_boxes)
                # if filtered_boxes.ndim == 1:
                #     filtered_boxes = filtered_boxes.reshape(1, -1)  # Convert (4,) to (1, 4)

                masks = segment_objects(frame, filtered_boxes*yolodownscale)
                # masks_temp = masks
                if masks is not None:
                    for mask in masks:
                        if mask is not None and mask.size > 0:
                            # If mask is boolean, convert it to uint8 (0 and 255)
                            mask_original = copy.deepcopy(mask)
                            if mask.dtype == bool:
                                mask = mask.astype(np.uint8) * 255

                            # If mask has 3 channels (RGB), convert to grayscale
                            if len(mask.shape) == 3:
                                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for cnt in contours:
                                # print("1")
                                if cv2.contourArea(cnt) > contours_area_threshold:  # Ignore small objects
                                    contours_area_temp = cv2.contourArea(cnt)    
                                    if contours_area < contours_area_temp:
                                        contours_area = contours_area_temp
                            print(f"Contour Area: {contours_area}")
                            
                            if contours_area > 0 :
                                if is_training_call:
                                    frame_copy = frame
                                else:
                                    frame_copy = frame.copy()
                                processed_mask = process_mask(mask_original, frame.shape)
                                width_height_ratio = detect_and_measure_edges(processed_mask, frame_copy)
                                pass
                                # processed_mask=cv2.resize(processed_mask, (int(width/downscale),int(height/downscale)))

                            
                        
                        #cv2.imshow("Mask_"+str(cam_id),processed_mask)
                        #processed_mask=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        
        frame=cv2.resize(frame, (int(width/downscale),int(height/downscale)))

        if not is_training_call:
            brush_id_name_ratio = brush_id_name+"_ratio"
            brush_ratio = brush_dimensions_json[brush_id_name_ratio]
            contours_area_tolarance = brush_dimensions * contours_area_tolarance_percentage / 100
            #cv2.imshow("Live Camera Feed"+str(cam_id),frame)
            if contours_area > contours_area_threshold:
                cb_dimensions_good = False
                if brush_dimensions - contours_area_tolarance <= contours_area <= brush_dimensions + contours_area_tolarance:
                    width_height_ratio_tolarance = brush_ratio*width_height_ratio_tolarance_percentage/100
                    dimension_error = False
                    if brush_ratio - width_height_ratio_tolarance <= width_height_ratio <= brush_ratio + width_height_ratio_tolarance:
                        cb_dimensions_good = True
                        ratio_error = False

        # print("Expected Dimension: "+str(brush_dimensions))
        # print("Contour Dimension: "+str(contours_area))
        log_info("___________________________________________")
        log_info("Camera index: "+str(camera_index))
        log_info("Brush ID: "+str(brush_id_name))
        log_info("Expected Dimension: "+str(brush_dimensions))
        log_info("Contour Dimension: "+str(contours_area))
        log_info("Area Tolarance: "+str(contours_area_tolarance)+" Diff: "+str(contours_area-brush_dimensions))
        if dimension_error:
            log_info("Dimension Error")
        else:
            log_info("Dimension Good")
        log_info("Expected Ratio: "+str(brush_ratio))
        log_info("Deteccted Ratio: "+str(width_height_ratio))
        log_info("Ratio Tolarance: "+str(width_height_ratio_tolarance)+" Diff: "+str(width_height_ratio-brush_ratio))
        if ratio_error:
            log_info("Ratio Error")
        else:
            log_info("Ratio Good")
    return frame, cb_dimensions_good, contours_area, width_height_ratio

def compare_dimension(actual_Value, brush_value):
    is_good_brush = False
    if actual_Value-dimension_tolarance < brush_value < actual_Value+dimension_tolarance:
        is_good_brush = True
    return is_good_brush

def frame_process_check_contour_dimension(camera_index, frame, brush_id_name, label_value_box, is_training_call):

    cb_dimensions_good = False
    dimension_error = True
    brush_length, brush_width, wire_length, wire_thickness, brush_height = (0,0,0,0,0)
    x1, y1, x2, y2 = label_value_box
    frame = frame[y1:y2, x1:x2]

    if is_training_call or brush_id_name in brush_dimensions_json:
        if is_top_camera(camera_index):
            brush_length, brush_width, wire_length, wire_thickness = dimension.check_contour_dimensions_top_camera(frame)
        else:
            brush_height = dimension.check_contour_dimensions_side_camera(frame)

        height, width, _ = frame.shape
        frame=cv2.resize(frame, (int(width/downscale),int(height/downscale)))

    if not is_training_call:
        brush_dimensions = brush_dimensions_json[brush_id_name]
        is_brush_length_good = False
        is_brush_width_good = False
        is_brush_height_good = False
        is_brush_wire_length_good = False
        is_brush_wire_thickness_good = False
        if brush_length > 0 and compare_dimension(brush_dimensions["length"], brush_length):
            is_brush_length_good = True
        if brush_width > 0 and compare_dimension(brush_dimensions["width"], brush_width):
            is_brush_width_good = True
        if brush_height > 0 and compare_dimension(brush_dimensions["height"], brush_height):
            is_brush_height_good = True
        if wire_thickness > 0 and compare_dimension(brush_dimensions["wire_thickness"], wire_thickness):
            is_brush_wire_length_good = True
        if wire_length > 0 and compare_dimension(brush_dimensions["wire_length"], wire_length):
            is_brush_wire_thickness_good = True
        if is_top_camera(camera_index):
            if is_brush_length_good and is_brush_width_good and is_brush_wire_length_good and is_brush_wire_thickness_good:
                cb_dimensions_good = True
                dimension_error = False
        else:
            if is_brush_height_good:
                cb_dimensions_good = True
                dimension_error = False

        log_info("___________________________________________")
        log_info("Camera index: "+str(camera_index))
        log_info("Brush ID: "+str(brush_id_name))
        log_info("Expected Dimensions L X W X H: {:.3f} X {:.3f} X {:.3f}".format(brush_dimensions["length"], brush_dimensions["width"], brush_dimensions["height"]))
        log_info("Expected Wire Length: "+str(brush_dimensions.get("wire_length","")))
        log_info("Expected Wire thickness : "+str(brush_dimensions.get("wire_thickness","")))
        log_info("Detected Dimensions L X W X H: {:.3f} X {:.3f} X {:.3f}".format(brush_length, brush_width, brush_height))
        log_info("Detected Wire Length: "+str(wire_length))
        log_info("Detected Wire thickness : "+str(wire_thickness))
        log_info("Dimension Tolarance: "+str(dimension_tolarance))
        if dimension_error:
            log_info("Dimension Error")
        else:
            log_info("Dimension Good")
    detected_brush_dimensions = {"brush_length":brush_length, "brush_width":brush_width, "brush_height": brush_height, "wire_length":wire_length, "wire_thickness": wire_thickness}
    return frame, cb_dimensions_good, detected_brush_dimensions


