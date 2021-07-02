import enum
from functools import cache

import cv2
import os
import time


class Moving(enum.Enum):
    X = 0
    Y = 1
    W = 2
    H = 3


class ObjectTracker:
    def __init__(self, filename, n_tracks=1):
        directory = os.path.join(os.path.dirname(__file__), 'haarcascade')
        self.classifier = cv2.CascadeClassifier(
            os.path.join(directory, filename))

        self.n_tracks = n_tracks
        self.last_positions = []
        self.ignore_frames = 0

    def ignore_next_frames(self, n):
        self.last_positions = []
        self.ignore_frames = n

    def update_position(self, img):
        if self.ignore_frames > 0:
            self.ignore_frames -= 1
            print(self.ignore_frames)
            return
        self.get_diff.cache_clear()
        if len(self.last_positions) >= self.n_tracks:
            self.last_positions.pop(0)
        positions = self.classifier.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if not len(positions):
            # Nothing detected. Move on.
            if len(self.last_positions):
                self.last_positions.pop(0)
            return False
        self.last_positions.append(positions)
        return True

    def highlight_objects(self, frame):
        if not len(self.last_positions):
            return
        for (x, y, w, h) in self.last_positions[-1]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    @cache
    def get_diff(self, initial_index=0, final_index=-1, object_index=0):
        if not len(self.last_positions):
            return 0, 0, 0, 0
        initial = self.last_positions[initial_index][object_index]
        final = self.last_positions[final_index][object_index]
        return tuple((a - b) / a for a, b in zip(initial, final))

    def is_moving(self, dimension, sensitivity, object_index=0):
        """
        Checks if the object is moving in the given dimension/axis.

        :param dimension: x=0; y=1; w=2; h=3
        :param sensitivity: Float percent difference threshold
        :param object_index: If more than 1 object is found, which one to use
        :return: boolean
        """
        dimension = dimension.value
        diffs = self.get_diff(initial_index=0, final_index=-1,
                              object_index=object_index)
        if sensitivity > 0:
            return diffs[dimension] > sensitivity
        elif sensitivity < 0:
            return diffs[dimension] < sensitivity

    def is_waving(self, sensitivity, object_index=0):
        if len(self.last_positions) < 3:
            return False
        mid = int(len(self.last_positions) / 2)
        to_mid_diff = self.get_diff(
            initial_index=0, final_index=mid, object_index=object_index)[2]
        to_end_diff = self.get_diff(
            initial_index=mid, final_index=-1, object_index=object_index)[2]
        if abs(to_mid_diff) < sensitivity or abs(to_end_diff) < sensitivity:
            return
        if to_mid_diff > sensitivity:
            return to_end_diff < -sensitivity
        if to_mid_diff < sensitivity:
            return to_end_diff > -sensitivity

    @property
    def total_frames(self):
        return len(self.last_positions)


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    face_tracker = ObjectTracker('frontalface.xml', 5)
    rpalm_tracker = ObjectTracker('rpalm.xml', 15)
    #lpalm_tracker = ObjectTracker('lpalm.xml', 15)
    palm_trackers = [rpalm_tracker,] #lpalm_tracker]
    profileface_tracker = ObjectTracker('profileface.xml', 20)

    while True:
        # Capture frame-by-frame
        time.sleep(0.05)
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_tracker.update_position(gray)
        rpalm_tracker.update_position(gray)
        #lpalm_tracker.update_position(gray)
        profileface_is_present = profileface_tracker.update_position(gray)

        face_tracker.highlight_objects(frame)
        rpalm_tracker.highlight_objects(frame)
        #lpalm_tracker.highlight_objects(frame)
        profileface_tracker.highlight_objects(frame)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # if face_tracker.is_moving(Moving.Y, -0.5):
        #     print("Moving down!")
        #     face_tracker.ignore_next_frames(25)
        # if face_tracker.is_moving(Moving.Y, +0.5):
        #     print("Moving up!")
        #     #face_tracker.ignore_next_frames(25)
        if face_tracker.is_moving(Moving.X, -0.5):
            os.system("cliclick kp:arrow-left")
            print("Moving left!")
            face_tracker.ignore_next_frames(25)
        if face_tracker.is_moving(Moving.X, +0.5):
            os.system("cliclick kp:arrow-right")
            print("Moving right!")
            face_tracker.ignore_next_frames(25)
        # if face_tracker.is_moving(Moving.W, -0.7):
        #     print("Moving forward!")
        #     #face_tracker.ignore_next_frames(25)
        # if face_tracker.is_moving(Moving.W, +0.5):
        #     print("Moving back!")
        #     #face_tracker.ignore_next_frames(25)
        if any(i.is_waving(0.1) for i in palm_trackers):
            os.system("cliclick kp:f2")
            print("Waving!")
        # if profileface_is_present:
        #     print("I see a profile")
        if profileface_tracker.is_moving(Moving.W, +0.3) and profileface_is_present:
            print("U turning!")
            os.system("cliclick kp:arrow-down")
            profileface_tracker.ignore_next_frames(100)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
