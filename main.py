import cv2

def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open the webcam.")
        return

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            cv2.imshow("AirCmd", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
