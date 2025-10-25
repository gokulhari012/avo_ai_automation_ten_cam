import cv2
import numpy as np
from scipy.signal import savgol_filter
import time
import numpy as np
import json
from logger import log_info, log_error

top_cam_pixel_per_mm = 31 # 30.8 to 31.2
side_cam_pixel_per_mm = 31 # 30.8 to 31.2

top_camera_threshold = 239
top_camera_min_area = 45000
top_camera_max_area = 55000

side_camera_threshold = 239
side_camera_min_area = 45000
side_camera_max_area = 55000

with open("configuration.json", "r") as file:
    json_data = json.load(file)
top_camera_threshold = json_data["top_camera_threshold"]
top_camera_min_area = json_data["top_camera_min_area"]
top_camera_max_area = json_data["top_camera_max_area"]

side_camera_threshold = json_data["side_camera_threshold"]
side_camera_min_area = json_data["side_camera_min_area"]
side_camera_max_area = json_data["side_camera_max_area"]

with open("configuration.json", "r") as file:
    json_data = json.load(file)
top_cam_pixel_per_mm = json_data["top_cam_pixel_per_mm"]
side_cam_pixel_per_mm = json_data["side_cam_pixel_per_mm"]

is_mesurement_in_mm_required = True

min_wire_thickness = 0.5 # 3 mm 
max_wire_thickness = 3 # 3 mm 

min_head_width = 2 # 8 mm
max_head_width = 8 # 8 mm

min_head_length = 2 # 8 mm
max_head_length = 8 # 8 mm

cam_position = "top"
brightness_contrast = 0
crop_width,crop_height=900,300

def nothing(x):
    pass

def get_white_centers(mask, min_white_pixels=30, max_white_pixels=50):
    """
    Get the center pixel from the white pixels in each vertical column of the image
    only if the number of white pixels is less than the specified limit.

    :param image: Input binary (black & white) image (numpy array).
    :param max_white_pixels: Maximum number of white pixels allowed in a column to consider it.
    :return: List of (x, y) coordinates of the center white pixels.
    """
    height, width = mask.shape
    center_points = []
    toggle_skip = 0
    for x in range(width):
        if toggle_skip==0:
            white_pixel_indices = np.where(mask[:, x] == 255)[0]
            
            # Only consider columns with fewer white pixels than the limit
            if min_white_pixels < len(white_pixel_indices) < max_white_pixels:
                center_y = int(np.median(white_pixel_indices))
                center_points.append((x, center_y))
        toggle_skip+=1
        if toggle_skip == 3:
            toggle_skip = 0

    return center_points


def filter_distant_points(center_points, max_distance=20):
    """
    Removes points where the distance between consecutive points is greater than max_distance.

    :param center_points: List of (x, y) coordinates.
    :param max_distance: Maximum allowed distance between consecutive points.
    :return: Filtered list of center points.
    """
    filtered_points = [center_points[0]] if center_points else []

    for i in range(1, len(center_points)):
        x1, y1 = filtered_points[-1]
        x2, y2 = center_points[i]
        
        # Calculate Euclidean distance
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Only add point if it's within the allowed distance
        if distance <= max_distance:
            filtered_points.append((x2, y2))

    return filtered_points

def calculate_line_length(center_points):
    """
    Calculate the total length of the line connecting all center points.

    :param center_points: List of (x, y) coordinates.
    :return: Total length of the connecting line.
    """
    length = 0.0
    for i in range(1, len(center_points)):
        x1, y1 = center_points[i - 1]
        x2, y2 = center_points[i]
        length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return length

def draw_center_line(image, center_points):
    """
    Draw a line connecting the center points on the image.

    :param image: Input grayscale or binary image.
    :param center_points: List of (x, y) coordinates of the center points.
    :return: Image with the centerline drawn.
    """
    # Convert to BGR for colored drawing
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i in range(1, len(center_points)):
        cv2.line(img_color, center_points[i - 1], center_points[i], (0, 0, 255), 1)  # Red line

    return img_color

def get_min_area_rect(image,contour):
       

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    return rect, box, contour

def align_and_crop(image, rect, padding=25):
    center, (width, height), angle = rect
    width, height = int(width), int(height)

    # Ensure the longer side is width for consistency
    if width < height:
        angle += 90
        width, height = height, width  # Swap to maintain correct aspect ratio

    # Get rotation matrix for the full image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the new image dimensions after rotation
    cos_a, sin_a = abs(rotation_matrix[0, 0]), abs(rotation_matrix[0, 1])
    new_w = int(image.shape[1] * cos_a + image.shape[0] * sin_a)
    new_h = int(image.shape[1] * sin_a + image.shape[0] * cos_a)


    # Adjust the transformation matrix to move the center
    rotation_matrix[0, 2] += (new_w // 2) - center[0]
    rotation_matrix[1, 2] += (new_h // 2) - center[1]

    # Rotate the full image
    # print()
    # print(new_w)
    # print(new_h)
    # print(rotation_matrix)

    # with open("output.txt", "w") as f:
    #     f.write(f"new_w: {new_w}\n")
    #     f.write(f"new_h: {new_h}\n")
    #     f.write(f"rotation_matrix: {rotation_matrix}\n")
        
    # new_w = 1979
    # new_h = 1188
    # #rotation_matrix = [[-9.91454296e-01, 1.30454506e-01, 1.94206965e+03], [-1.30454506e-01, -9.91454296e-01, 1.28489018e+03]]
    # rotation_matrix = np.array([[-9.98339935e-01, 5.75966497e-02, 1.77902293e+03], [-5.75966497e-02, -9.98339935e-01, 1.03327927e+03]], dtype=np.float32) 
    
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    # Extract the aligned object using sub-pixel extraction
    cropped = cv2.getRectSubPix(rotated, (width + 2 * padding, height + 2 * padding), (new_w // 2, new_h // 2))
    
    return cropped

def process_frame(frame, threshold, min_area, max_area):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bin_fail, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
   
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    if not filtered_contours:
        return frame
    
    largest_contour = max(filtered_contours, key=cv2.contourArea)

    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask,largest_contour

def get_value_in_pixel(mm_value):
    pixel_per_mm = 0
    if cam_position == "top":
        pixel_per_mm = top_cam_pixel_per_mm
    elif cam_position == "side":
        pixel_per_mm = side_cam_pixel_per_mm
    return mm_value*pixel_per_mm

def get_value_in_mm(pixel_value):
    pixel_per_mm = 0
    if cam_position == "top":
        pixel_per_mm = top_cam_pixel_per_mm
    elif cam_position == "side":
        pixel_per_mm = side_cam_pixel_per_mm
    return pixel_value/pixel_per_mm

def get_brush_length(vertical_white_pixel_counts):
    brush_length_pixel_count = [pixel_count for pixel_count in vertical_white_pixel_counts if get_value_in_pixel(min_head_length) <= pixel_count <= get_value_in_pixel(max_head_length)] 
    brush_length_average = 0
    if brush_length_pixel_count:
        brush_length_average = sum(brush_length_pixel_count) / len(brush_length_pixel_count)
    return brush_length_average

def get_brush_width(horizontal_white_pixel_counts):
    brush_width_pixel_count = [pixel_count for pixel_count in horizontal_white_pixel_counts if get_value_in_pixel(min_head_width) <= pixel_count <= get_value_in_pixel(max_head_width)] 
    brush_width_average = 0
    if brush_width_pixel_count:
        brush_width_average = sum(brush_width_pixel_count) / len(brush_width_pixel_count)
    return brush_width_average

def get_wire_thickness(vertical_white_pixel_counts):
    wire_thickness_pixel_count = [pixel_count for pixel_count in vertical_white_pixel_counts if get_value_in_pixel(min_wire_thickness) <= pixel_count <= get_value_in_pixel(max_wire_thickness)] 
    wire_thickness_average = 0
    if wire_thickness_pixel_count:
        wire_thickness_average = sum(wire_thickness_pixel_count) / len(wire_thickness_pixel_count)
    return wire_thickness_average

def process_mask(mask, contour, camera_side):
    min_rect= get_min_area_rect(mask,contour)
    if len(min_rect)>2:
        rect, box, contour = min_rect
        cropped = align_and_crop(mask, rect)
        #axis 0 is horizontal lines and axis 1 is vertical lines.
        horizontal_white_pixel_counts = np.sum(cropped == 255, axis=1)
        vertical_white_pixel_counts = np.sum(cropped == 255, axis=0)
        # print(horizontal_white_pixel_counts)
        if camera_side == "top":
            brush_length = get_brush_length(vertical_white_pixel_counts)
            brush_width = get_brush_width(horizontal_white_pixel_counts)
            wire_thickness = get_wire_thickness(vertical_white_pixel_counts)

            centers = get_white_centers(cropped, min_white_pixels = get_value_in_pixel(min_wire_thickness), max_white_pixels = get_value_in_pixel(max_wire_thickness))
            filtered_centers = filter_distant_points(centers, max_distance=10)
            wire_line_length = calculate_line_length(filtered_centers)
            output_img = draw_center_line(cropped, filtered_centers)
            unit = "px"
            if is_mesurement_in_mm_required:
                unit = "mm"
                brush_length = get_value_in_mm(brush_length)
                brush_width = get_value_in_mm(brush_width)
                wire_thickness = get_value_in_mm(wire_thickness)
                wire_line_length = get_value_in_mm(wire_line_length)
            # cv2.putText(output_img, "Total brush Length: {:.3f} mm".format(total_length/adv), (int(cropped.shape[1]*0.1),int(cropped.shape[0]*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(output_img, "Total brush Width: {:.3f} mm".format(total_height/adv), (int(cropped.shape[1]*0.1),int(cropped.shape[0]*0.17)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # output_img=cv2.resize(output_img,(crop_width,crop_height))
            text_x_axis = 0.1
            text_y_axis = 0.5
            text_y_axis_increment = 0.07
            cv2.putText(output_img, "Head Length: {:.3f} {}".format(brush_length, unit), (int(cropped.shape[1]*text_x_axis),int(cropped.shape[0]*(text_y_axis+text_y_axis_increment*0))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(output_img, "Head Width: {:.3f} {}".format(brush_width, unit), (int(cropped.shape[1]*text_x_axis),int(cropped.shape[0]*(text_y_axis+text_y_axis_increment*1))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(output_img, "2 wire L: {:.3f} mm".format(w_l/adv), (150, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(output_img, "Wire Length: {:.3f} {}".format(wire_line_length, unit), (int(cropped.shape[1]*text_x_axis),int(cropped.shape[0]*(text_y_axis+text_y_axis_increment*2))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(output_img, "Wire Thickness: {:.3f} {}".format(wire_thickness, unit), (int(cropped.shape[1]*text_x_axis),int(cropped.shape[0]*(text_y_axis+text_y_axis_increment*3))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(output_img, "Total brush Length: {:.3f} mm".format(t_l/adv), (int(cropped.shape[1]*0.1),int(cropped.shape[0]*0.64)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return output_img, brush_length, brush_width, wire_line_length, wire_thickness        
        else:
            brush_height = get_brush_length(vertical_white_pixel_counts)
            unit = "px"
            if is_mesurement_in_mm_required:
                unit = "mm"
                brush_height = get_value_in_mm(brush_height)                
            text_x_axis = 0.1
            text_y_axis = 0.5
            text_y_axis_increment = 0.07
            cv2.putText(cropped, "Head Height: {:.3f} {}".format(brush_height, unit), (int(cropped.shape[1]*text_x_axis),int(cropped.shape[0]*(text_y_axis+text_y_axis_increment*0))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return cropped, brush_height        
            
def on_toggle(val):
    global brightness_contrast
    brightness_contrast = val  # Update toggle state
    # print(f"Toggle State: {'ON' if brightness_contrast else 'OFF'}")

def check_contour_dimensions_top_camera(frame):
    brush_length, brush_width, wire_line_length, wire_thickness = (0,0,0,0)
    processed = process_frame(frame, top_camera_threshold, top_camera_min_area, top_camera_max_area)
    if(len(processed)<3):
        mask, contour = processed
        cpd, brush_length, brush_width, wire_line_length, wire_thickness = process_mask(mask, contour, "top")
        cv2.imshow("Proceesed image : Top", cpd)
    return brush_length, brush_width, wire_line_length, wire_thickness

def check_contour_dimensions_side_camera(frame):
    brush_height = 0
    processed = process_frame(frame, side_camera_threshold, side_camera_min_area, side_camera_max_area)
    if(len(processed)<3):
        mask, contour = processed
        cpd, brush_height = process_mask(mask, contour, "side")
        cv2.imshow("Proceesed image : Side", cpd)
    return brush_height
