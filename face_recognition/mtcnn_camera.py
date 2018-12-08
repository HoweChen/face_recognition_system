import face_recognition
import cv2
import redis
import numpy as np
from mtcnn.mtcnn import MTCNN


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)

def get_random_RGB_color():
    return tuple(np.random.choice(range(256), size=3))


if __name__ == '__main__':
    # connect to the redis service
    redis_pool = redis.ConnectionPool()
    r = redis.Redis(connection_pool=redis_pool)

    obama_image_file = "examples/obama.jpg"
    biden_image_file = "examples/biden.jpg"
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file(obama_image_file)
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    str_obama_face_encoding = obama_face_encoding.tostring()

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file(biden_image_file)
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    str_biden_face_encoding = biden_face_encoding.tostring()

    # make sure that the list is empty then append the base data
    if r.rpushx("known_face_encodings", str_obama_face_encoding) == 0 and r.rpushx("known_face_names",
                                                                                   "Barack Obama".encode("utf-8")) == 0:
        r.rpush("known_face_encodings", str_obama_face_encoding, str_biden_face_encoding)
        r.rpush("known_face_names", "Barack Obama".encode("utf_8"), "Joe Biden".encode("utf-8"))

    # Create arrays of known face encodings and their names
    # known_face_encodings = [
    #     obama_face_encoding,
    #     biden_face_encoding
    # ]
    # known_face_names = [
    #     "Barack Obama",
    #     "Joe Biden"
    # ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    detector = MTCNN()

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            # face_locations = face_recognition.face_locations(rgb_small_frame)
            detect_result = detector.detect_faces(rgb_small_frame)[0]["box"]
            face_locations = [tuple(
                    [detect_result[1]+40, detect_result[0] + detect_result[2] + 3,
                     detect_result[1] + detect_result[-1] + 10,
                     detect_result[0]])]
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            # print(f"Answer: {face_locations}")
            # print(f"Raw: {detect_result}")
            # print(f"Modified: {face_locations_two}")
            face_names = []
            best_point_list = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                true_or_false, points = face_recognition.compare_faces(
                        list(map(lambda x: np.frombuffer(x), r.lrange("known_face_encodings", 0, -1))), face_encoding,
                        tolerance=0.4)
                name = "Unknown"
                best_point = None

                # If a match was found in known_face_encodings, just use the first one.
                if True in true_or_false and len(true_or_false) == len(points):
                    best_point = min(points)
                    winner_index = points.index(best_point)
                    print(winner_index)
                    name = r.lrange("known_face_names", 0, -1)[winner_index].decode("utf-8")
                best_point_list.append(best_point)
                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), temp_name, temp_best_point in zip(face_locations, face_names, best_point_list):
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

        # Hit '1' on the keyboard to input the new face into the database!
        if cv2.waitKey(1) & 0xFF == ord('1'):

            try:
                r.rpushx("known_face_encodings", face_encodings[0].tostring())
                r.rpushx("known_face_names", "Yuhao Chen")
                print("Success with:", end=" ")
                print(r.lrange("known_face_names", 0, -1))
            except Exception as e:
                continue

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
