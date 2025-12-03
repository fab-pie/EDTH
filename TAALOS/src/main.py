import cv2
import time
from src.core.uav_tracking_system import UAVTrackingSystem
from src.core.motor import Motor
from utils.bbox import get_bbox_center

FOV_X = 81
RES_W = 1920
RES_H = 1080
RATIO_DEG_PER_PIXEL = FOV_X / RES_W
X_CENTER = RES_W / 2

# Scanning parameters (when no drone detected)
SCAN_ANGLE = 1000  # Angle to scan (degrees)
SCAN_DELAY = 20.0   # Delay between scan movements (seconds)

motor = Motor("/dev/tty.usbserial-0001")  # Replace with your serial port
motor.connect()

track_sys = UAVTrackingSystem(
    model_path="../models/yolov11n-UAV-finetune.pt",
    conf_threshold=0.5,
    device="cpu",
)


# webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")


try:
    previous_pose_x = 0.0

    # Scanning state variables
    last_scan_time = time.time()
    scan_direction = 1  # 1 for positive, -1 for negative
    scanning_mode = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Cannot read frame from webcam")
            break

        # Process frame
        annotated_frame, bbox_xyxy = track_sys.process_frame(frame)

        if bbox_xyxy is not None:
            motor.send_speed(15)
            # TRACKING MODE: Drone detected
            scanning_mode = False

            pos_x, pos_y = get_bbox_center(bbox_xyxy)

            f_pos_x = pos_x * 0.4 + previous_pose_x * 0.6
            previous_pose_x = f_pos_x

            # draw cross at the center of the bbox (convert to int for OpenCV)
            cv2.drawMarker(annotated_frame, (int(pos_x), int(pos_y)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            pixels_error_x = f_pos_x - X_CENTER
            angle_error_x = pixels_error_x * RATIO_DEG_PER_PIXEL

            angle_command = angle_error_x * 1.0  # P gain = 1.0
            print(f"Tracking: {angle_command:.2f}°")

            motor.send_angle(angle_command)

            # Reset scan timer
            last_scan_time = time.time()

        # else:
        #     motor.send_speed(3)
        #     # SCANNING MODE: No drone detected
        #     current_time = time.time()
        #
        #     if not scanning_mode:
        #         # Just entered scanning mode
        #         print("No drone detected - Starting scan mode")
        #         scanning_mode = True
        #         last_scan_time = current_time
        #         motor.send_angle(scan_direction * SCAN_ANGLE)
        #         print(f"Scanning: {scan_direction * SCAN_ANGLE}°")
        #
        #     elif current_time - last_scan_time >= SCAN_DELAY:
        #         # Time to reverse direction
        #         scan_direction *= -1
        #         motor.send_angle(scan_direction * SCAN_ANGLE)
        #         print(f"Scanning: {scan_direction * SCAN_ANGLE}°")
        #         last_scan_time = current_time

        # Display frame
        cv2.imshow('TAALOS', annotated_frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    motor.send_angle(0)
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")
