import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math
import json

def align_object(image, threshold_value=200, output_size=None):
    """
    Aligns the main object in a white-background grayscale or color image using PCA.
    
    Args:
        image (np.ndarray): Input grayscale or BGR image.
        threshold_value (int): Threshold to separate object from background.
        output_size (tuple): Optional (width, height) to resize output.

    Returns:
        np.ndarray: Rotated and aligned image.
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Threshold to create binary mask (assuming dark object on white background)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Find contours (use external to ignore noise)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No object found.")
        return image

    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute orientation using PCA
    data_pts = largest_contour[:, 0, :].astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(data_pts, mean=None, maxComponents=2)

    # Calculate angle between the major axis and the x-axis
    angle_rad = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
    angle_deg = np.degrees(angle_rad)

    # Rotate image to align the object
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_LINEAR, borderValue=(255,255,255) if len(image.shape)==3 else 255)

    # Optional: crop or resize to remove borders
    if output_size:
        rotated = cv2.resize(rotated, output_size)

    return rotated

def remove_small_dust(image, area_threshold=50):
    """
    Removes small dark specks (dust) from a white background grayscale image.

    Args:
        image (np.ndarray): Grayscale image (uint8 or float32).
        area_threshold (int): Maximum area to consider a speck (in pixels).

    Returns:
        np.ndarray: Cleaned grayscale image.
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img = image.copy()

    # Threshold to isolate dark spots (dust)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  # Dust is dark (<200)

    # Find connected components (potential dust particles)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Create a mask to keep only non-dust areas
    dust_mask = np.ones_like(img, dtype=np.uint8) * 255

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= area_threshold:
            dust_mask[labels == i] = 0  # Mask out this dust speck

    # Inpaint the dust specks using surrounding pixels
    mask = cv2.bitwise_not(dust_mask)  # Inpainting mask must be white on defects
    cleaned = cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    return cleaned

# === Load and prepare image ===
image_path = './templates/template_1.bmp'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
gray = remove_small_dust(gray, area_threshold=50000).astype(np.float32)
#gray = align_object(gray, threshold_value=200).astype(np.float32)

height, width = gray.shape

# === Storage ===
horizontal_lines = []
vertical_lines = []
annotations = []
h_vals_def=[]
v_vals_def=[]
fig, ax = plt.subplots()
im = ax.imshow(gray, cmap='gray', aspect='auto')
plt.title("Press H/V to add horizontal/vertical path. Drag to move. Click 'Measure' to analyze.")
plt.axis("on")

def save_line(event=None):
    # Extract coordinates
    horizontal_data = [list(zip(line.line.get_xdata(), line.line.get_ydata())) for line in horizontal_lines]
    vertical_data = [list(zip(line.line.get_xdata(), line.line.get_ydata())) for line in vertical_lines]

    # Save to JSON
    with open("./templates/template_1.json", "w") as f:
        json.dump({
            "horizontal_lines": horizontal_data,
            "vertical_lines": vertical_data,
            "h_vals_def":h_vals_def,
            "v_vals_def":v_vals_def
        }, f)

def load_line(event=None):

    with open("./templates/template_1.json", "r") as f:
        data = json.load(f)

    # Recreate horizontal lines
    for coords in data.get("horizontal_lines", []):
        print(coords)
        line = ax.axhline(coords[1][1], color='cyan', linestyle='--')
        horizontal_lines.append(DraggableLine(line, orientation='horizontal'))
        fig.canvas.draw()

    # Recreate vertical lines
    for coords in data.get("vertical_lines", []):
        print(coords)
        line = ax.axvline(coords[0][0], color='magenta', linestyle='--')
        vertical_lines.append(DraggableLine(line, orientation='vertical'))
        fig.canvas.draw()


def remove_last_line(event=None):
    if horizontal_lines or vertical_lines:
        if horizontal_lines and (not vertical_lines or horizontal_lines[-1].line.axes.get_figure().canvas.get_renderer() is not None):
            line = horizontal_lines.pop().line
        else:
            line = vertical_lines.pop().line
        line.remove()
        fig.canvas.draw()

# === Subpixel Edge Detection ===
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

# === Draggable Line ===
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

# === Add Lines ===
def add_horizontal(y):
    line = ax.axhline(y, color='cyan', linestyle='--')
    horizontal_lines.append(DraggableLine(line, orientation='horizontal'))
    fig.canvas.draw()

def add_vertical(x):
    line = ax.axvline(x, color='magenta', linestyle='--')
    vertical_lines.append(DraggableLine(line, orientation='vertical'))
    fig.canvas.draw()

# === Measurement Logic ===
def measure_all(event=None):
    for ann in annotations:
        ann.remove()
    annotations.clear()
    h_vals_def.clear()
    v_vals_def.clear()
    threshold = 128
    mm_per_pixel = 0.0075#0.02038 # # Adjust this to your calibration
    delta =50 # offset for reference lines
    num_parallel = 31 # Must be odd (e.g., 7 = center ±3)
    half = num_parallel // 2

    for hline in horizontal_lines:
        y_center = int(hline.line.get_ydata()[0])
        if delta + half < y_center < height - (delta + half):
            all_distances = []

            for offset in range(-half, half + 1):
                y = y_center + offset

                profile = gray[y, :]
                edges = detect_subpixel_edges(profile, threshold)

                top_profile = gray[y - delta, :]
                bot_profile = gray[y + delta, :]
                top_edges = detect_subpixel_edges(top_profile, threshold)
                bot_edges = detect_subpixel_edges(bot_profile, threshold)

                for i in range(0, min(len(edges), len(top_edges), len(bot_edges)) - 1, 2):
                    x0, x1 = edges[i], edges[i + 1]
                    x0_t, x1_t = top_edges[i], top_edges[i + 1]
                    x0_b, x1_b = bot_edges[i], bot_edges[i + 1]

                    dx = ((x1_b - x1_t) + (x0_b - x0_t)) / 2
                    dy = 2 * delta
                    angle_rad = math.atan2(dx, dy)

                    dist_px = abs(x1 - x0) * abs(math.cos(angle_rad))
                    dist_mm = dist_px * mm_per_pixel
                    all_distances.append(dist_mm)

            if all_distances:
                data = np.array(all_distances)
                mean = np.mean(data)
                std = np.std(data)

                if std > 0:
                    z_scores = np.abs((data - mean) / std)
                    filtered = data[z_scores < 0.6]  # Keep only values within 2 std deviations
                else:
                    filtered = data  # No variation; use all

                if len(filtered) > 0:
                    avg_mm = np.mean(filtered)
                    h_vals_def.append(round(avg_mm,3))
                    annotation = ax.annotate(f"{avg_mm:.2f} mm", xy=(width // 2, y_center),
                                         color='red', fontsize=9, ha='center', va='bottom')
                    annotations.append(annotation)

    for vline in vertical_lines:
        x_center = int(vline.line.get_xdata()[0])
        if delta + half < x_center < width - (delta + half):
            all_distances = []

            for offset in range(-half, half + 1):
                x = x_center + offset

                profile = gray[:, x]
                edges = detect_subpixel_edges(profile, threshold)

                left_profile = gray[:, x - delta]
                right_profile = gray[:, x + delta]
                left_edges = detect_subpixel_edges(left_profile, threshold)
                right_edges = detect_subpixel_edges(right_profile, threshold)

                for i in range(0, min(len(edges), len(left_edges), len(right_edges)) - 1, 2):
                    y0, y1 = edges[i], edges[i + 1]
                    y0_l, y1_l = left_edges[i], left_edges[i + 1]
                    y0_r, y1_r = right_edges[i], right_edges[i + 1]

                    dy = ((y1_r - y1_l) + (y0_r - y0_l)) / 2
                    dx = 2 * delta
                    angle_rad = math.atan2(dy, dx)

                    dist_px = abs(y1 - y0) * abs(math.cos(angle_rad))
                    dist_mm = dist_px * mm_per_pixel
                    all_distances.append(dist_mm)

            if all_distances:
                data = np.array(all_distances)
                mean = np.mean(data)
                std = np.std(data)

                if std > 0:
                    z_scores = np.abs((data - mean) / std)
                    filtered = data[z_scores < 2]  # Keep only values within 2 std deviations
                else:
                    filtered = data  # No variation; use all

                if len(filtered) > 0:
                    avg_mm = np.mean(filtered)
                    v_vals_def.append(round(avg_mm,3))
                annotation = ax.annotate(f"{avg_mm:.2f} mm", xy=(x_center, height // 2),
                                         color='lime', fontsize=9, ha='left', va='center')
                annotations.append(annotation)

    fig.canvas.draw()

# === Keyboard Control ===
def on_key(event):
    if event.inaxes != ax: return
    if event.key.lower() == 'h':
        add_horizontal(event.ydata)
    elif event.key.lower() == 'v':
        add_vertical(event.xdata)
    elif event.key.lower() == 'r':
        remove_last_line()

fig.canvas.mpl_connect('key_press_event', on_key)

# === save Line Button ===
ax_load = plt.axes([0.20, 0.02, 0.20, 0.045])
btn_load = Button(ax_load, 'load template')
btn_load.on_clicked(load_line)

# === save Line Button ===
ax_save = plt.axes([0.40, 0.02, 0.20, 0.045])
btn_save = Button(ax_save, 'save template')
btn_save.on_clicked(save_line)

# === Remove Last Line Button ===
ax_remove = plt.axes([0.60, 0.02, 0.20, 0.045])
btn_remove = Button(ax_remove, 'Remove Line')
btn_remove.on_clicked(remove_last_line)

# === Measure Button ===
ax_button = plt.axes([0.82, 0.02, 0.12, 0.045])
btn = Button(ax_button, 'Measure')
btn.on_clicked(measure_all)

plt.show()
