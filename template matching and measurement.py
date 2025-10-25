output_dir = "matched_crops"
import os
os.makedirs(output_dir, exist_ok=True)


    for point_info in points_list:
        point = point_info[0]
        angle = point_info[1]
        scale = point_info[2]

        x, y = point
        w_scaled = int(width * scale / 100)
        h_scaled = int(height * scale / 100)

        # Crop the region from the image
        crop = img_rgb[y:y + h_scaled, x:x + w_scaled]

        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue  # Skip if crop is invalid

        # Create a white canvas and paste the crop at center
        canvas_size = max(w_scaled, h_scaled)
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        x_offset = (canvas_size - w_scaled) // 2
        y_offset = (canvas_size - h_scaled) // 2
        canvas[y_offset:y_offset + h_scaled, x_offset:x_offset + w_scaled] = crop

        # Rotate the canvas to align the crop
        image_center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        aligned_crop = cv2.warpAffine(canvas, rot_mat, (canvas.shape[1], canvas.shape[0]),
                                      flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        # Crop tightly again after alignment (optional)
        aligned_gray = cv2.cvtColor(aligned_crop, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(aligned_gray, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x_al, y_al, w_al, h_al = cv2.boundingRect(contours[0])
            final_crop = aligned_crop[y_al:y_al + h_al, x_al:x_al + w_al]
        else:
            final_crop = aligned_crop

        # Save the final crop
        save_path = os.path.join(output_dir, f"match_{match_id:03}.png")
        cv2.imwrite(save_path, cv2.cvtColor(final_crop, cv2.COLOR_RGB2BGR))  # Convert back to BGR to save
        print(f"Saved: {save_path}")
        match_id += 1
