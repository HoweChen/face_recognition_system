import face_recognition
import cv2
import redis
import numpy as np
from mtcnn.mtcnn import MTCNN
from enum import Enum
import base64
from typing import AnyStr


class ImagePath(Enum):
    OBAMA_IMAGE_FILE = "../code_base/examples/obama.jpg"
    BIDEN_IMAGE_FILE = "../code_base/examples/biden.jpg"


class Camera:
    def __init__(self, mode="HOG"):
        # connect to the redis service
        self.mode = mode
        self.redis_pool = redis.ConnectionPool()
        self.r = redis.Redis(connection_pool=self.redis_pool)
        self.detector = MTCNN() if mode == "MTCNN" else None
        self.video_capture = None

        self.data_validation()

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.best_point_list = []

    def serve(self):

        self.video_capture = cv2.VideoCapture(0)

        process_this_frame = True

        for frame in self.frame_fetch(self.video_capture):
            # Grab a single frame of video
            ret, frame = frame

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which code_base uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                self.face_detection(rgb_small_frame)
                self.face_matching()

            process_this_frame = not process_this_frame

            # Display the results
            self.render_boxes(frame)
            yield frame

        # Release handle to the webcam
        self.video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def frame_fetch(video_capture):
        while True:
            yield video_capture.read()

    def face_detection(self, input_frame):
        # Find all the faces and face encodings in the current frame of video
        if self.mode == "MTCNN" and self.detector is not None:
            try:
                detect_result = [face["box"] for face in self.detector.detect_faces(input_frame)]
            except IndexError:
                # which means the detector doesn't find the faces, then directly show the image without box
                detect_result = None
            if detect_result is not None:
                # the detector does find the faces
                self.face_locations = [tuple(
                        [single_face[1], single_face[0] + single_face[2],
                         single_face[1] + single_face[-1],
                         single_face[0]]) for single_face in detect_result]
                self.face_encodings = face_recognition.face_encodings(input_frame, self.face_locations)

        if self.mode == "HOG":
            self.face_locations = face_recognition.face_locations(input_frame)
            self.face_encodings = face_recognition.face_encodings(input_frame, self.face_locations)

    def face_matching(self):

        self.face_names = []
        self.best_point_list = []

        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            true_or_false, points = face_recognition.compare_faces(
                    list(map(lambda x: np.frombuffer(x), self.r.lrange("known_face_encodings", 0, -1))),
                    face_encoding,
                    tolerance=0.4)
            name = "Unknown"
            best_point = None

            # If a match was found in known_face_encodings, just use the first one.
            if True in true_or_false and len(true_or_false) == len(points):
                best_point = min(points)
                winner_index = points.index(best_point)
                print(winner_index)
                name = self.r.lrange("known_face_names", 0, -1)[winner_index].decode("utf-8")
            self.face_names.append(name)
            self.best_point_list.append(best_point)

    def render_boxes(self, frame):
        # Display the results
        for (top, right, bottom, left), temp_name, temp_best_point in zip(self.face_locations, self.face_names,
                                                                          self.best_point_list):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            if temp_best_point is not None:
                cv2.putText(frame, f"{temp_name}: {temp_best_point:.3f}%", (left + 6, bottom - 6), font, 0.9,
                            (255, 255, 255), 1)
            else:
                cv2.putText(frame, f"{temp_name}", (left + 6, bottom - 6), font, 0.9,
                            (255, 255, 255), 1)

    def data_validation(self):
        # Load a sample picture and learn how to recognize it.
        obama_image = face_recognition.load_image_file(ImagePath.OBAMA_IMAGE_FILE.value)
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        str_obama_face_encoding = obama_face_encoding.tostring()

        # Load a second sample picture and learn how to recognize it.
        biden_image = face_recognition.load_image_file(ImagePath.BIDEN_IMAGE_FILE.value)
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
        str_biden_face_encoding = biden_face_encoding.tostring()

        # make sure that the list is empty then append the code_base data
        if self.r.rpushx("known_face_encodings", str_obama_face_encoding) == 0 and self.r.rpushx("known_face_names",
                                                                                                 "Barack Obama".encode(
                                                                                                         "utf-8")) == 0:
            self.r.rpush("known_face_encodings", str_obama_face_encoding, str_biden_face_encoding)
            self.r.rpush("known_face_names", "Barack Obama".encode("utf_8"), "Joe Biden".encode("utf-8"))


# the following export API methods are split from the class Camera

def string_to_image(input_str: str) -> np.ndarray:
    decoded_data = base64.b64decode(input_str)
    np_data = np.fromstring(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    return image


def image_to_str(input_image) -> AnyStr:
    retval, buf = cv2.imencode(".jpg", input_image)
    output_str = base64.b64encode(buf).decode()
    return output_str


def data_validation(redis_connection):
    obama_image = face_recognition.load_image_file(ImagePath.OBAMA_IMAGE_FILE.value)
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    str_obama_face_encoding = obama_face_encoding.tostring()

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file(ImagePath.BIDEN_IMAGE_FILE.value)
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    str_biden_face_encoding = biden_face_encoding.tostring()

    # make sure that the list is empty then append the code_base data
    if redis_connection.rpushx("known_face_encodings", str_obama_face_encoding) == 0 and redis_connection.rpushx(
            "known_face_names",
            "Barack Obama".encode(
                    "utf-8")) == 0:
        redis_connection.rpush("known_face_encodings", str_obama_face_encoding, str_biden_face_encoding)
        redis_connection.rpush("known_face_names", "Barack Obama".encode("utf_8"), "Joe Biden".encode("utf-8"))


def face_matching():
    pass


def face_detection():
    pass


def render_box():
    pass


def single_serve(input_image_str, redis_connection, mode="MTCNN"):
    data_validation(redis_connection=redis_connection)
    tmp_image_from_str = string_to_image(input_image_str)
    tmp_output_image = render_box()
    output_image_str = image_to_str(tmp_output_image)

    return output_image_str


if __name__ == '__main__':
    instance = Camera(mode="MTCNN")
    instance.serve()
