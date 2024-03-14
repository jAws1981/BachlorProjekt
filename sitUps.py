import cv2
import mediapipe as mp
import winsound

class SitupCounter:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.count = 0
        self.is_up = False

    def count_situps(self, landmarks):
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y
        left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y
        left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y

        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y
        right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y
        right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y

        avg_shoulder = (left_shoulder + right_shoulder) / 2
        avg_hip = (left_hip + right_hip) / 2
        avg_knee = (left_knee + right_knee) / 2

        if avg_shoulder < avg_hip and avg_hip < avg_knee:
            self.is_up = True
        elif avg_shoulder > avg_hip and avg_hip > avg_knee and self.is_up:
            self.count += 1
            self.is_up = False
            # Play beep sound
            winsound.Beep(1000, 200)  # Frequenz: 1000 Hz, Dauer: 200 ms

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                self.count_situps(results.pose_landmarks.landmark)

                cv2.putText(
                    image,
                    f"Situps: {self.count}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

            cv2.namedWindow("Sit-up Counter", cv2.WINDOW_NORMAL)
            cv2.imshow("Sit-up Counter", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    situp_counter = SitupCounter()
    situp_counter.run()
