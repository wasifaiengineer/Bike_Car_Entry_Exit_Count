import cv2
from bike_tracker import ObjectCounter

# Mouse callback for debugging (optional)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

# Load video
cap = cv2.VideoCapture('D:/Computer_vision_projects/Bike_counting/data/CLIP_21042025_084237_C1 (2).mp4')

# Define line region (update these points as needed for your use case)
region_points = [(1546, 1504), (1754, 954)]

# Create ObjectCounter instance
counter = ObjectCounter(
    region=region_points,
    model="D:/Computer_vision_projects/Car_counting_colour_brand/yolo12n.pt",
    show_in=True,
    show_out=True,
    line_width=2
)

# Optional: initialize region if needed
if hasattr(counter, 'initialize_region'):
    counter.initialize_region()

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # Ensure frame is valid before processing
    if frame is None or frame.size == 0:
        continue

    # Process frame
    counter_results = counter.process(frame)

    # Resize for display
    display_frame = cv2.resize(counter_results.plot_im, (1020, 500))
    cv2.imshow("RGB", display_frame)

    print(f"[Frame {frame_count}] IN: {counter_results.in_count} | OUT: {counter_results.out_count} | Total: {counter_results.total_tracks}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
