import face_recognition
import cv2
import redis
import numpy as np
from mtcnn.mtcnn import MTCNN
# from multiprocessing.dummy import Pool
from enum import Enum
from imutils.video import FPS
from multiprocessing import Process, Queue
import time


class ImagePath(Enum):
    OBAMA_IMAGE_FILE = "examples/obama.jpg"
    BIDEN_IMAGE_FILE = "examples/biden.jpg"


class Camera:
    def __init__(self, mode="HOG"):
        # connect to the redis service
        self.mode = mode
        self.redis_pool = redis.ConnectionPool()
        self.r = redis.Redis(connection_pool=self.redis_pool)
        self.detector = MTCNN() if mode == "MTCNN" else None
        self.video_capture = None
        self.data_validation()
        self.process_this_frame = False
        self.quit_all_processes = False

        # # Initialize some variables
        # self.face_locations = []
        # self.face_encodings = []
        # self.face_names = []
        # self.best_point_list = []

        # multiprocessing usage setup
        self.captured_queue = Queue()
        self.detected_queue = Queue()
        self.recognized_queue = Queue()
        self.result_queue = Queue()

        # FPS monitor setup
        self.fps = FPS()

    def serve(self):

        # capture_process = Process(target=self.capturer)
        detection_process = Process(target=self.face_detection)
        recognition_process = Process(target=self.face_matching)
        rendering_process = Process(target=self.render_boxes)
        show_process = Process(target=self.show_result)

        # processes = [capture_process, detection_process, recognition_process,rendering_process,show_process]
        processes = [detection_process, recognition_process,
                     rendering_process]

        self.video_capture = cv2.VideoCapture(0)

        for process in processes:
            process.start()

        self.fps.start()

        for ret, frame in self.frame_fetch(self.video_capture):

            # print(frame)
            if ret is False and frame is None:
                self.captured_queue.put(None)  # put None to the queue to tell the sub process quit the job
                break
            else:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which code_base uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time
                if self.process_this_frame is True:
                    self.captured_queue.put(rgb_small_frame)
                    # self.face_detection(rgb_small_frame)
                    # self.face_matching()

                self.process_this_frame = not self.process_this_frame

        # when the program run till here means that the break point has been triggered
        # shutdown all worker processes
        for process in processes:
            try:
                process.join()
            except Exception as e:
                print(e)
                continue

        # Release handle to the webcam
        self.video_capture.release()
        cv2.destroyAllWindows()

    def frame_fetch(self, video_capture):
        while True:
            if self.quit_all_processes is False:
                yield video_capture.read()  # which will return True and the image frame
            else:
                yield (False, None)

    #
    # def capturer(self):
    #     self.fps.start()
    #     self.video_capture = cv2.VideoCapture(0)
    #
    #     for frame in self.frame_fetch(self.video_capture):
    #         ret, frame = frame
    #         # Grab a single frame of video
    #         if frame is None:
    #             # self.captured_queue.put(None)
    #             # break
    #             continue
    #         else:
    #             # Resize frame of video to 1/4 size for faster face recognition processing
    #             small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #
    #             # Convert the image from BGR color (which OpenCV uses) to RGB color (which code_base uses)
    #             rgb_small_frame = small_frame[:, :, ::-1]
    #
    #             # Only process every other frame of video to save time
    #             if self.process_this_frame is True:
    #                 self.captured_queue.put(rgb_small_frame)
    #                 # self.face_detection(rgb_small_frame)
    #                 # self.face_matching()
    #             self.process_this_frame = not self.process_this_frame
    #
    #         # # Display the results
    #         # self.render_boxes(frame)
    #         #
    #         # # simulate the data input process when there is nothing in the database
    #         # # Hit '1' on the keyboard to input the new face into the database!
    #         # if cv2.waitKey(1) & 0xFF == ord('1'):
    #         #
    #         #     try:
    #         #         self.r.rpushx("known_face_encodings", self.face_encodings[0].tostring())
    #         #         self.r.rpushx("known_face_names", "Yuhao Chen")
    #         #         print("Success with:", end=" ")
    #         #         print(self.r.lrange("known_face_names", 0, -1))
    #         #     except Exception as e:
    #         #         print(e)
    #         #
    #         # # Display the resulting image
    #         # cv2.imshow('Video', frame)
    #         # self.fps.update()
    #         #
    #         # # Hit 'q' on the keyboard to quit!
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #
    #     # self.fps.stop()
    #     # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #     # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    #     #
    #     # # Release handle to the webcam
    #     # self.video_capture.release()
    #     # cv2.destroyAllWindows()

    def face_detection(self):

        while True:
            is_queue_empty = self.captured_queue.empty()

            if is_queue_empty is True:
                continue
            else:
                # fetch the input frame from the queue
                input_frame = self.captured_queue.get()

                if input_frame is not None:
                    # Find all the faces and face encodings in the current frame of video

                    face_locations= []
                    face_encodings = []

                    if self.mode == "MTCNN" and self.detector is not None:
                        try:
                            detect_result = [
                                face["box"] for face in self.detector.detect_faces(input_frame)]
                        except IndexError:
                            # which means the detector doesn't find the faces, then directly show the image without box
                            detect_result = None

                        if detect_result is not None:
                            # the detector does find the faces
                            face_locations = [tuple(
                                [single_face[1], single_face[0] + single_face[2],
                                 single_face[1] + single_face[-1],
                                 single_face[0]]) for single_face in detect_result]
                            face_encodings = face_recognition.face_encodings(
                                input_frame, face_locations)

                    if self.mode == "HOG":
                        face_locations = face_recognition.face_locations(
                            input_frame)
                        face_encodings = face_recognition.face_encodings(
                            input_frame, face_locations)

                    self.detected_queue.put(
                        (input_frame, face_locations, face_encodings))
                else:
                    self.detected_queue.put((None, None, None))
                    break

    def face_matching(self):

        while True:

            is_queue_empty = self.detected_queue.empty()

            if is_queue_empty is True:
                continue
            else:
                # fetch the input_frame,face_locations,face_encodings from detected_queue
                input_frame, face_locations, face_encodings = self.detected_queue.get()

                if (input_frame, face_locations, face_encodings) is not (None, None, None):

                    face_names = []
                    best_point_list = []

                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        true_or_false, points = face_recognition.compare_faces(
                            list(map(lambda x: np.frombuffer(x), self.r.lrange(
                                "known_face_encodings", 0, -1))),
                            face_encoding,
                            tolerance=0.4)
                        name = "Unknown"
                        best_point = None

                        # If a match was found in known_face_encodings, just use the first one.
                        if True in true_or_false and len(true_or_false) == len(points):
                            best_point = min(points)
                            winner_index = points.index(best_point)
                            print(winner_index)
                            name = self.r.lrange(
                                "known_face_names", 0, -1)[winner_index].decode("utf-8")
                        face_names.append(name)
                        best_point_list.append(best_point)

                    self.recognized_queue.put(
                        (input_frame, face_locations, face_encodings, face_names, best_point_list))
                else:
                    self.recognized_queue.put((None, None, None, None, None))
                    break

    def render_boxes(self):
        while True:
            is_queue_empty = self.recognized_queue.empty()
            if is_queue_empty is True:
                continue
            else:
                input_frame, face_locations, face_encodings, face_names, best_point_list = self.recognized_queue.get()
                if (input_frame, face_locations, face_encodings, face_names, best_point_list) is not (
                        None, None, None, None, None):

                    # Display the results
                    for (top, right, bottom, left), temp_name, temp_best_point in zip(face_locations, face_names,
                                                                                      best_point_list):
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        # Draw a box around the face
                        cv2.rectangle(input_frame, (left, top),
                                      (right, bottom), (0, 0, 255), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(input_frame, (left, bottom - 35),
                                      (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        if temp_best_point is not None:
                            cv2.putText(input_frame, f"{temp_name}: {temp_best_point:.3f}%", (left + 6, bottom - 6),
                                        font, 0.9,
                                        (255, 255, 255), 1)
                        else:
                            cv2.putText(input_frame, f"{temp_name}", (left + 6, bottom - 6), font, 0.9,
                                        (255, 255, 255), 1)

                    self.result_queue.put(input_frame)
                    # print(input_frame)

                else:
                    self.result_queue.put(None)
                    break

    def show_result(self):

        while True:
            is_queue_empty = self.result_queue.empty()
            if is_queue_empty is True:
                continue
            else:
                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit_all_processes = True
                    break
                show_frame = self.result_queue.get()
                if show_frame is not None:
                    cv2.imshow('Video', show_frame)
                    self.fps.update()
                else:
                    break

        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

    def data_validation(self):
        # Load a sample picture and learn how to recognize it.
        obama_image = face_recognition.load_image_file(
            ImagePath.OBAMA_IMAGE_FILE.value)
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        str_obama_face_encoding = obama_face_encoding.tostring()

        # Load a second sample picture and learn how to recognize it.
        biden_image = face_recognition.load_image_file(
            ImagePath.BIDEN_IMAGE_FILE.value)
        biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
        str_biden_face_encoding = biden_face_encoding.tostring()

        # make sure that the list is empty then append the code_base data
        if self.r.rpushx("known_face_encodings", str_obama_face_encoding) == 0 and self.r.rpushx("known_face_names",
                                                                                                 "Barack Obama".encode(
                                                                                                     "utf-8")) == 0:
            self.r.rpush("known_face_encodings",
                         str_obama_face_encoding, str_biden_face_encoding)
            self.r.rpush("known_face_names", "Barack Obama".encode(
                "utf_8"), "Joe Biden".encode("utf-8"))


if __name__ == '__main__':
    cam = Camera(mode="HOG")
    # cam = Camera(mode="MTCNN")
    cam.serve()
