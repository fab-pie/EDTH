import cv2
from src.core.uav_tracking_system import UAVTrackingSystem


def main():
    """
    Main function to run UAV detection/tracking from webcam.
    """
    # Initialize tracking system
    system = UAVTrackingSystem(
        model_path="../models/yolov11n-UAV-finetune.pt",
        conf_threshold=0.6,
        device="cpu",
    )

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Starting webcam UAV detection/tracking. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame from webcam")
                break

            # Process frame
            annotated_frame, bbox_xyxy = system.process_frame(frame)

            # Display frame
            cv2.imshow('UAV Detection / Tracking - Webcam', annotated_frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam detection stopped.")


if __name__ == "__main__":
    main()
