import cv2

# Global list to store clicked points
region_points = []
scale = 0.5  # Reduce size to 50%

# Mouse callback function (adjusted for scaled image)
def draw_region(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale back coordinates to original resolution
        region_points.append((int(x / scale), int(y / scale)))

# Load sample frame from video (first frame)
cap = cv2.VideoCapture('D:\\Computer_vision_projects\\Bike_counting\\data\\CLIP_21042025_084237_C1 (2).mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to load frame from video.")
    exit()

# Setup window and scaled frame
clone = frame.copy()
display_frame = cv2.resize(clone, (0, 0), fx=scale, fy=scale)

cv2.namedWindow("Draw Region")
cv2.setMouseCallback("Draw Region", draw_region)

print("🖱 Click to draw polygon points. Press 's' to save. Press 'q' to quit.")

while True:
    temp_frame = cv2.resize(clone.copy(), (0, 0), fx=scale, fy=scale)

    # Draw scaled lines as you click
    if len(region_points) > 1:
        for i in range(len(region_points) - 1):
            pt1 = tuple(int(v * scale) for v in region_points[i])
            pt2 = tuple(int(v * scale) for v in region_points[i + 1])
            cv2.line(temp_frame, pt1, pt2, (0, 255, 0), 2)
        pt_last = tuple(int(v * scale) for v in region_points[-1])
        pt_first = tuple(int(v * scale) for v in region_points[0])
        cv2.line(temp_frame, pt_last, pt_first, (0, 255, 0), 2)

    # Draw circles
    for pt in region_points:
        scaled_pt = tuple(int(v * scale) for v in pt)
        cv2.circle(temp_frame, scaled_pt, 5, (0, 0, 255), -1)

    cv2.imshow("Draw Region", temp_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("✅ Saved Region Points:")
        print(region_points)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
