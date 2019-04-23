import face_recognition
import cv2
import redis
import numpy as np
from mtcnn.mtcnn import MTCNN
import multiprocessing.dummy as T
import multiprocessing as P
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from code_base.frame_size import FrameSize
from code_base.StegaStamp.stegastamp import StegaStamp


class ImagePath(Enum):
    OBAMA_IMAGE_FILE = "examples/obama.jpg"
    BIDEN_IMAGE_FILE = "examples/biden.jpg"


class ImageOption(Enum):
    SHRINK_RATE = 0.25
    ENLARGE_RATE = 4


class Instance:
    def __init__(self, mode="HOG", f_e_m="NORMAL", watermark=False):
        print("Camera instance is initializing...")
        self.video_capture: cv2.VideoCapture
        self.mode = mode
        self.f_e_m = f_e_m  # detect which feature extraction method
        self.redis_pool = redis.ConnectionPool()
        try:
            # connect to the redis service
            self.r = redis.Redis(connection_pool=self.redis_pool)
        except ConnectionError as e:
            print(e)
        self.detector = MTCNN() if mode == "MTCNN" else None
        self.watermark = watermark
        if watermark is True:
            self.stegastmp = StegaStamp(mode="ENCODE")
        else:
            self.stegastmp: StegaStamp
        # self.data_validation()

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.best_point_list = []

    def serve(self):
        self.video_capture = cv2.VideoCapture(0)

        process_this_frame = False
        register_process = T.Process(target=self.register_to_db)
        pool = T.Pool(processes=None)

        for frame, timer_start in self.frame_fetch(self.video_capture):

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Grab a single frame of video
            ret, frame = frame

            # processing_process = T.Process(target=self.process_image, args=(frame,))
            # Only process every other frame of video to save time
            if process_this_frame:
                pool.apply_async(self.process_image, args=(frame,))
                # self.process_image(frame)

            process_this_frame = not process_this_frame

            # Hit '1' on the keyboard to input the new face into the database!
            if cv2.waitKey(1) & 0xFF == ord('1'):

                try:
                    register_process.start()
                    # register_process.join()
                except Exception as e:
                    print(e)

            # Display the results
            self.render_boxes(frame)
            # Display the resulting image
            cv2.imshow('Video', frame)

            # time the performance
            timer_diff = Decimal(str((cv2.getTickCount() - timer_start) / cv2.getTickFrequency())).quantize(
                Decimal("0.000"),
                rounding=ROUND_HALF_UP)
            print(f"[Time Info]: Time from input to output: {timer_diff}s")

        # Release handle to the webcam
        # pool.close()
        # pool.join()
        self.video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def frame_fetch(video_capture: cv2.VideoCapture):
        while True:
            yield (video_capture.read(), cv2.getTickCount())

    def register_to_db(self):
        # find the largest face with its index according to the face_locations
        index_max, location_max = max(enumerate(self.face_locations), key=lambda x: abs(x[1][1] - x[1][0])
                                                                                    * abs(
            x[1][2] - x[1][1]))
        # face_index = self.face_locations.index(max_size_face)
        print(f"face index: {index_max}")
        new_face_encoding_string = self.face_encodings[index_max].tostring()
        print(self.face_encodings[index_max])

        self.r.rpush("known_face_encodings", new_face_encoding_string)
        self.r.rpush("known_face_names", "Yuhao Chen".encode("utf_8"))
        print("Success with:", end=" ")
        print(self.r.lrange("known_face_names", 0, -1))

    def process_image(self, frame):
        rgb_small_frame = self.frame_to_rgb_samll_frame(frame=frame)
        self.face_detection(rgb_small_frame)
        if self.watermark is True:
            watermarked_frames = self.face_locations_to_stegastamp_size(rgb_small_frame)
            self.face_extraction(watermarked_frames)
        else:
            self.face_extraction(rgb_small_frame)
        self.face_matching()

    @staticmethod
    def frame_to_rgb_samll_frame(frame, image_size="LARGE"):
        if image_size == "LARGE":
            # only when image is large
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame: np.ndarray = cv2.resize(frame, (0, 0), fx=ImageOption.SHRINK_RATE.value,
                                                 fy=ImageOption.SHRINK_RATE.value)
            # height, width = small_frame.shape[:2]
            # print(f"{height}x{width}") # print the size of resized frame
        else:
            small_frame = frame

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        return rgb_small_frame

    def face_detection(self, input_frame):

        # Find all the faces and face encodings in the current frame of vid
        face_locations = []
        if self.mode == "MTCNN":
            if self.detector is None:
                self.detector = MTCNN()
            else:
                try:
                    detect_result = [face["box"] for face in self.detector.detect_faces(input_frame)]
                except IndexError:
                    # which means the detector doesn't find the faces, then directly show the image without box
                    detect_result = None

                if detect_result is not None:
                    # the detector does find the faces
                    face_locations = [tuple(
                        [single_face[1], single_face[0] + single_face[2],
                         single_face[1] + single_face[-1],
                         single_face[0]]) for single_face in detect_result]

        elif self.mode == "HOG" or self.mode == "CNN":
            tmp = face_recognition.face_locations(input_frame, model=self.mode.lower())
            # if self.watermark is False:
            #     face_locations = tmp
            # else:
            #     face_locations = raw_to_watermark_square(tmp)
            face_locations = tmp
        else:
            raise Exception("Invalid face detection method")

        self.face_locations = face_locations

    def face_locations_to_stegastamp_size(self, rgb_small_frame):
        batched_raw_faces = []
        for (top, right, bottom, left) in self.face_locations:
            top = int(top)
            right = int(right)
            bottom = int(bottom)
            left = int(left)
            # slice_frame = rgb_small_frame[left:top, right:bottom]
            slice_frame = rgb_small_frame[top:bottom, left:right]
            batched_raw_faces.append(slice_frame)
        watermarked_frames = self.stegastmp.encode(batched_raw_faces, asecret="Hello")
        return watermarked_frames

    def face_extraction(self, input_frame, num_jitters=1):
        # feature extraction method
        face_encodings = []
        if self.f_e_m == "NORMAL":
            if self.watermark is False:
                face_encodings = face_recognition.face_encodings(input_frame, self.face_locations, num_jitters)
            else:
                # if watermark is True, then the input_frame is a list of encoded frames
                for frame in input_frame:
                    # height, width = frame.shape[:2]
                    # print(f"{height}x{width}") # print the size of resized frame
                    face_encodings.append(face_recognition.face_encodings(frame, [(0, 400, 400, 0)], num_jitters)[0])
        elif self.f_e_m == "FACENET":
            pass
        else:
            raise Exception("Invalid feature extraction method")
        self.face_encodings = face_encodings

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
            if any(true_or_false) and len(true_or_false) == len(points):
                winner_index, best_point = min(enumerate(points), key=lambda x: x[1])
                best_point = str(Decimal(best_point).quantize(Decimal("0.00"), rounding=ROUND_HALF_UP))
                print(f"[Match Info]: Winner: {winner_index}, best point: {best_point}")
                name = self.r.lrange("known_face_names", 0, -1)[winner_index - 1].decode("utf-8")
            self.face_names.append(name)
            self.best_point_list.append(best_point)

    def render_boxes(self, frame):
        # Display the results
        for (top, right, bottom, left), temp_name, temp_best_point in zip(self.face_locations, self.face_names,
                                                                          self.best_point_list):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= ImageOption.ENLARGE_RATE.value
            right *= ImageOption.ENLARGE_RATE.value
            bottom *= ImageOption.ENLARGE_RATE.value
            left *= ImageOption.ENLARGE_RATE.value

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            dot_coordinate = ((left + right) // 2, (top + bottom) // 2)
            cv2.circle(frame, (dot_coordinate[0], dot_coordinate[1]), 5, (0, 0, 255), -1)

            font = cv2.FONT_HERSHEY_DUPLEX
            if temp_best_point is not None:
                cv2.putText(frame, f"{temp_name}: {temp_best_point}%", (left + 6, bottom - 6), font, 0.9,
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

        # make sure that the list is empty then append the facial data
        if self.r.rpushx("known_face_encodings", str_obama_face_encoding) == 0 and self.r.rpushx("known_face_names",
                                                                                                 "Barack Obama".encode(
                                                                                                     "utf-8")) == 0:
            self.r.rpush("known_face_encodings", str_obama_face_encoding, str_biden_face_encoding)
            self.r.rpush("known_face_names", "Barack Obama".encode("utf_8"), "Joe Biden".encode("utf-8"))

    def image_to_db(self, filepath: str, name: str, image_size, num_jitters):
        frame = cv2.imread(filepath)
        rgb_small_frame = self.frame_to_rgb_samll_frame(frame=frame, image_size=image_size)
        self.face_detection(rgb_small_frame)
        self.face_extraction(rgb_small_frame, num_jitters=num_jitters)
        # get only one person at a time
        face_encoding = self.face_encodings[0]
        face_encoding_str = face_encoding.tostring()
        # make sure that the list is empty then append the code_base data
        # if self.r.rpushx("known_face_encodings", face_encoding_str) == 0 and self.r.rpushx("known_face_names",
        #                                                                                    name.encode(
        #                                                                                        "utf-8")) == 0:
        self.r.rpush("known_face_encodings", face_encoding_str)
        self.r.rpush("known_face_names", name.encode("utf_8"))


if __name__ == '__main__':
    cv2.useOptimized()
    instance = Instance(mode="MTCNN", f_e_m="NORMAL", watermark=True)
    # instance = Instance(mode="MTCNN", f_e_m="NORMAL")
    # instance = Instance(mode="HOG", f_e_m="NORMAL")
    # instance = Instance(mode="CNN", f_e_m="NORMAL")  # very slow, not recommended
    instance.serve()
