import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import time
import os
from InvariantTM import template_crop, invariant_match_template
from pypylon import pylon
import math
import json
import tkinter as tk
from tkinter import filedialog

#plt.ion()
#fig, ax = plt.subplots(figsize=(5,9))

plt.title("Measurements")

h_vals_def=[]
v_vals_def=[]

auto = True
trigger=True
auto = False
trigger=False

annotations = []
tolerance = 0.015 
def align_image(image):
    ds=7
    image_ds = cv2.resize(image, (image.shape[1] // ds, image.shape[0] // ds))
    gray = cv2.cvtColor(image_ds, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i=0
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 100000//ds:
            
            continue
        #print(i)
        i=i+1
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w > h:
            angle += 90
            w, h = h, w

        center = (int(cx)*ds, int(cy)*ds)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderValue=(255, 255, 255))
        
        return rotated , True
    return image, False

def remove_small_dust(image, area_threshold=5000):
    ds=5
    image_ds = cv2.resize(image, (image.shape[1] // ds, image.shape[0] // ds))
    img = image_ds.astype(np.uint8)

    # Step 1: Threshold the image to create a binary mask of dust particles
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Connected components to find dust regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Step 3: Create mask only where dust (small area) is present
    mask = np.zeros_like(img, dtype=np.uint8)
    for i in range(1, num_labels):  # Skip label 0 (background)
        if stats[i, cv2.CC_STAT_AREA] <= area_threshold//ds:
            mask[labels == i] = 255

    # Step 4: Inpaint the detected small regions (if any)
    if np.count_nonzero(mask) > 0:
        mask=cv2.resize(mask, (image.shape[1], image.shape[0]))
        output=cv2.inpaint(image, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
        #out_prv= cv2.resize(output,(output.shape[1]//4,output.shape[0]//4)).astype(np.uint8)
        #cv2.imshow("output_dust_removel", out_prv)
        return output
    else:
        return img  # No dust detected, return original


def match_and_crop(img_rgb, template_rgb, width, height, output_dir):
    ds = 25
    flipped = False

    # Downscale for faster matching
    img_rgb_ds = cv2.resize(img_rgb, (img_rgb.shape[1] // ds, img_rgb.shape[0] // ds))
    template_rgb_ds = cv2.resize(template_rgb, (template_rgb.shape[1] // ds, template_rgb.shape[0] // ds))

    def try_match(template):
        return invariant_match_template(
            rgbimage=img_rgb_ds,
            rgbtemplate=template,
            method="TM_CCOEFF_NORMED",
            matched_thresh=0.70,
            rot_range=[0, 360],
            rot_interval=90,
            scale_range=[99, 101],
            scale_interval=1,
            rm_redundant=True,
            minmax=True
        )

    # Attempt match
    points_list = try_match(template_rgb_ds)

    # Try flipped template if not found
    if not points_list:
        template_rgb_ds = cv2.flip(template_rgb_ds, 1)
        points_list = try_match(template_rgb_ds)
        flipped = True

    if not points_list:
        return img_rgb, False  # No match found

    for point, angle, scale, _ in points_list:
        x, y = point[0] * ds, point[1] * ds
        w_scaled = int(width * scale / 100)
        h_scaled = int(height * scale / 100)

        # Ensure crop bounds do not exceed image size
        x_end = min(x + w_scaled, img_rgb.shape[1])
        y_end = min(y + h_scaled, img_rgb.shape[0])
        crop = img_rgb[y:y_end, x:x_end]

        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue

        # Rotate crop
        center = (crop.shape[1] // 2, crop.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        aligned_crop = cv2.warpAffine(
            crop,
            rot_mat,
            (crop.shape[1], crop.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=(255, 255, 255)
        )

        return aligned_crop, True

    return img_rgb, False  # fallback in case no usable crop is created

def detect_subpixel_edges(profile, threshold=128):
    indices = np.where(np.diff(np.sign(profile - threshold)))[0]
    subpixel_edges = []
    for idx in indices:
        x0, x1 = idx, idx + 1
        y0, y1 = profile[x0], profile[x1]
        if y1 != y0:
            subpixel = x0 + (threshold - y0) / (y1 - y0)
            subpixel_edges.append(subpixel)
    return subpixel_edges

# === DraggableLine and GUI Logic ===
class DraggableLine:
    def __init__(self, line, orientation='horizontal'):
        self.line = line
        self.orientation = orientation
        self.press = None
        self.cidpress = line.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        contains, _ = self.line.contains(event)
        if not contains: return
        self.press = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.line.axes: return
        dx = event.xdata - self.press[0] if event.xdata else 0
        dy = event.ydata - self.press[1] if event.ydata else 0
        if self.orientation == 'horizontal':
            ydata = self.line.get_ydata()
            self.line.set_ydata([y + dy for y in ydata])
        elif self.orientation == 'vertical':
            xdata = self.line.get_xdata()
            self.line.set_xdata([x + dx for x in xdata])
        self.press = (event.xdata, event.ydata)
        self.line.figure.canvas.draw()

    def on_release(self, event):
        self.press = None

# === Image Processing and Measurement GUI ===
def analyze_frame_cv(
    image,
    template_data,
    mm_per_pixel=0.0075,
    delta=50,
    num_parallel=11,
    threshold=128,
    tolerance=0.01,
    scale=1.5  # <- increase for larger text/lines
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = remove_small_dust(gray, area_threshold=50000).astype(np.float32)
    output = image.copy()

    h_vals, v_vals = [], []
    h_vals_def = template_data.get("h_vals_def", [])
    v_vals_def = template_data.get("v_vals_def", [])

    font_scale = 0.5 * scale
    font_thickness = int(1 * scale)
    line_thickness = int(1 * scale)
    spacing = int(18 * scale)
    table_bg_color = (220, 220, 220)

    # Horizontal measurement
    for i, coords in enumerate(template_data.get("horizontal_lines", [])):
        y_center = int(coords[1][1])
        all_distances = []
        for offset in range(-num_parallel // 2, num_parallel // 2 + 1):
            y = y_center + offset
            if y - delta < 0 or y + delta >= gray.shape[0]:
                continue
            profile = gray[y, :]
            top_profile = gray[y - delta, :]
            bot_profile = gray[y + delta, :]
            edges = detect_subpixel_edges(profile, threshold)
            top_edges = detect_subpixel_edges(top_profile, threshold)
            bot_edges = detect_subpixel_edges(bot_profile, threshold)

            for j in range(0, min(len(edges), len(top_edges), len(bot_edges)) - 1, 2):
                x0, x1 = edges[j], edges[j + 1]
                x0_t, x1_t = top_edges[j], top_edges[j + 1]
                x0_b, x1_b = bot_edges[j], bot_edges[j + 1]

                dx = ((x1_b - x1_t) + (x0_b - x0_t)) / 2
                dy = 2 * delta
                angle_rad = math.atan2(dx, dy)
                dist_px = abs(x1 - x0) * abs(math.cos(angle_rad))
                all_distances.append(dist_px * mm_per_pixel)

        if all_distances:
            data = np.array(all_distances)
            mean = np.mean(data)
            std = np.std(data)
            filtered = data[np.abs((data - mean) / std) < 2] if std > 0 else data
            if len(filtered):
                avg_mm = round(np.mean(filtered), 3)
                h_vals.append(avg_mm)
                cv2.putText(output, f"{avg_mm:.2f} mm", (10, y_center - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                cv2.line(output, (0, y_center), (output.shape[1], y_center),
                         (255, 255, 0), line_thickness)

    # Vertical measurement
    for i, coords in enumerate(template_data.get("vertical_lines", [])):
        x_center = int(coords[0][0])
        all_distances = []
        for offset in range(-num_parallel // 2, num_parallel // 2 + 1):
            x = x_center + offset
            if x - delta < 0 or x + delta >= gray.shape[1]:
                continue
            profile = gray[:, x]
            left_profile = gray[:, x - delta]
            right_profile = gray[:, x + delta]
            edges = detect_subpixel_edges(profile, threshold)
            left_edges = detect_subpixel_edges(left_profile, threshold)
            right_edges = detect_subpixel_edges(right_profile, threshold)

            for j in range(0, min(len(edges), len(left_edges), len(right_edges)) - 1, 2):
                y0, y1 = edges[j], edges[j + 1]
                y0_l, y1_l = left_edges[j], left_edges[j + 1]
                y0_r, y1_r = right_edges[j], right_edges[j + 1]

                dy = ((y1_r - y1_l) + (y0_r - y0_l)) / 2
                dx = 2 * delta
                angle_rad = math.atan2(dy, dx)
                dist_px = abs(y1 - y0) * abs(math.cos(angle_rad))
                all_distances.append(dist_px * mm_per_pixel)

        if all_distances:
            data = np.array(all_distances)
            mean = np.mean(data)
            std = np.std(data)
            filtered = data[np.abs((data - mean) / std) < 2] if std > 0 else data
            if len(filtered):
                avg_mm = round(np.mean(filtered), 3)
                v_vals.append(avg_mm)
                cv2.putText(output, f"{avg_mm:.2f} mm", (x_center + 5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                cv2.line(output, (x_center, 0), (x_center, output.shape[0]),
                         (255, 0, 255), line_thickness)

    # Deviation Table Overlay
    y_offset = output.shape[0] - int((len(h_vals) + len(v_vals) + 2) * spacing)
    cv2.rectangle(output, (5, y_offset - 10),
                  (output.shape[1] - 5, output.shape[0] - 5), table_bg_color, -1)

    table_entries = []
    for i, (d, c) in enumerate(zip(h_vals_def, h_vals)):
        dev = round(c - d, 2)
        color = (0, 255, 0) if abs(dev) <= tolerance else (0, 0, 255)
        text = f"H{i+1}: D:{d:.2f} C:{c:.2f} Δ:{dev:+.2f}"
        table_entries.append((text, color))

    for i, (d, c) in enumerate(zip(v_vals_def, v_vals)):
        dev = round(c - d, 2)
        color = (0, 255, 0) if abs(dev) <= tolerance else (0, 0, 255)
        text = f"V{i+1}: D:{d:.2f} C:{c:.2f} Δ:{dev:+.2f}"
        table_entries.append((text, color))

    for i, (text, color) in enumerate(table_entries):
        y = y_offset + (i + 1) * spacing
        cv2.putText(output, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, font_thickness)

    return output

def measure_features(lines, gray, delta, threshold, mm_per_pixel, direction='horizontal'):
    results = []
    num_parallel = 11
    half = num_parallel // 2
    height, width = gray.shape

    for line in lines:
        coord = int(line.line.get_ydata()[0] if direction == 'horizontal' else line.line.get_xdata()[0])
        limit = height if direction == 'horizontal' else width
        if delta + half >= coord or coord >= limit - (delta + half):
            continue

        distances = []
        for offset in range(-half, half + 1):
            y = coord + offset if direction == 'horizontal' else None
            x = coord + offset if direction == 'vertical' else None

            profile = gray[y, :] if direction == 'horizontal' else gray[:, x]
            edges = detect_subpixel_edges(profile, threshold)

            prof1 = gray[y - delta, :] if direction == 'horizontal' else gray[:, x - delta]
            prof2 = gray[y + delta, :] if direction == 'horizontal' else gray[:, x + delta]
            edges1 = detect_subpixel_edges(prof1, threshold)
            edges2 = detect_subpixel_edges(prof2, threshold)

            for i in range(0, min(len(edges), len(edges1), len(edges2)) - 1, 2):
                p0, p1 = edges[i], edges[i + 1]
                p0_1, p1_1 = edges1[i], edges1[i + 1]
                p0_2, p1_2 = edges2[i], edges2[i + 1]

                d_other = ((p1_2 - p1_1) + (p0_2 - p0_1)) / 2
                main_axis = 2 * delta
                angle = math.atan2(d_other, main_axis)

                dist_px = abs(p1 - p0) * abs(math.cos(angle))
                distances.append(dist_px * mm_per_pixel)

        if distances:
            d = np.array(distances)
            mean, std = np.mean(d), np.std(d)
            filtered = d[np.abs((d - mean) / std) < 2] if std > 0 else d
            if len(filtered):
                results.append(round(np.mean(filtered), 3))

    return results

    
def match(image,template_rgb):
    output_dir = "matched_crops"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        img_bgr =image
        if img_bgr is None:
            print("No image captured from camera. Exiting.")
            break

        img_bgr,flg = align_image(img_bgr)
        if flg:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            #prv_match= cv2.resize(img_bgr,(img_bgr.shape[1]//4,img_bgr.shape[0]//4))
            #cv2.imshow("Captured RGB Image", prv_match)
            #template_bgr_prv= cv2.resize(template_bgr,(template_bgr.shape[1]//4,template_bgr.shape[0]//4))
            #cv2.imshow("templateImage", template_bgr_prv )
            
            cropped_template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
            height, width = cropped_template_gray.shape

            
            matched,match_flg=match_and_crop(img_rgb, template_rgb, width, height, output_dir)
            if match_flg:
                
                return matched,match_flg
            else:
                return img_bgr,match_flg

        else:
            return img_bgr,flg


# === Live Camera Integration ===
def run_camera_interface():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    template_bgr = plt.imread('./templates/template_1.bmp')
    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)
    with open("./templates/template_1.json", "r") as f:
        template_data = json.load(f)

    if trigger:
        camera.TriggerSelector.SetValue('FrameStart')
        camera.TriggerMode.SetValue('On')
        camera.TriggerSource.SetValue('Line1')

    camera.ExposureAuto.SetValue('Off')
    camera.ExposureTime.SetValue(300)
    camera.GainAuto.SetValue('Off')
    camera.Gain.SetValue(30.0)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    print("Press 'c' to capture current frame as target image, 'q' to quit.")

    while True:
        if not camera.IsGrabbing():
            break
        
        grab_result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
        #print(grab_result.GrabSucceeded())
        if grab_result.IsValid():
            start = time.time()
            frame = converter.Convert(grab_result).GetArray()
            prv= cv2.resize(frame,(frame.shape[1]//10,frame.shape[0]//10))
            cv2.imshow("Live Preview - Press 'c' to capture", prv)
            if auto:
                final_frame,flg=match(frame,template_rgb)
                if auto and flg:
                    #final_prv= cv2.resize(final_frame,(final_frame.shape[1]//4,final_frame.shape[0]//4))
                    #cv2.imshow("match", final_prv)
                    out=analyze_frame_cv(final_frame, template_data,scale=4)
                    out_prv= cv2.resize(out,(out.shape[1]//4,out.shape[0]//4))
                    cv2.imshow("output", out_prv)
                    end = time.time()
                    print(f"Total runtime of the program is {end - start} seconds")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                start = time.time()
                final_frame,flg=match(frame,template_rgb)
                #end = time.time()
                #print(f"Total runtime of the match is {end - start} seconds")
                if flg:
                    #final_prv= cv2.resize(final_frame,(final_frame.shape[1]//4,final_frame.shape[0]//4))
                    #cv2.imshow("match", final_prv)
                    #cv2.imwrite("./templates/template_1.bmp", final_frame)
                    #start = time.time()
                    out=analyze_frame_cv(final_frame, template_data,scale=4)
                    out_prv= cv2.resize(out,(out.shape[1]//4,out.shape[0]//4))
                    cv2.imshow("output", out_prv)
                    end = time.time()
                    print(f"Total runtime of the analyze is {end - start} seconds")
 
            elif key == ord('q'):
                camera.StopGrabbing()
                camera.Close()
                cv2.destroyAllWindows()
                return None

        grab_result.Release()
    

# === Run Program ===
if __name__ == "__main__":
    run_camera_interface()
