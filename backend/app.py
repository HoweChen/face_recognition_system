from cv2.cv2 import VideoCapture
import flask
from flask import Flask, render_template, Response
import cv2
from redis import ConnectionPool
from enum import Enum
from flask_camera_service import Camera
from flask_socketio import SocketIO, join_room, leave_room, send, emit
import redis
from typing import List

app: Flask = Flask(__name__)
# vc: VideoCapture = cv2.VideoCapture(0)
socketio: SocketIO = SocketIO(app)
socketio.camera_instances_dict = {}
redis_pool: ConnectionPool = redis.ConnectionPool()


class Constant(Enum):
    JPEG_BASE64_STRING_HEADER_LENGTH = 23  # "data:image/jpeg;base64," length is 23


@app.route('/')
def hello_world():
    return 'Hello World!'


def gen():
    """Video streaming generator function."""
    camera_instance = Camera(mode="MTCNN")
    # while True:
    #     ret, frame = vc.read()
    #     if not ret:
    #         continue
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode(".jpg",frame)[1].tostring() + b'\r\n')
    for frame in camera_instance.serve():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode(".jpg", frame)[1].tostring() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on("message")
def handle_message(message):
    print(message)


@socketio.on("join")
def on_join(data):
    user_name = data["userName"]
    room = data["room"]
    print(room)
    join_room(room, sid=room)  # in here the room string and request.sid are the same
    send(f"{user_name} has entered the room: {room}.", room=room)
    # print(socketio.room_users)


@socketio.on("leave")
def on_leave(data):
    user_name = data["userName"]
    room = data["room"]
    leave_room(room, sid=room)
    send(f"{user_name} has left the room: {room}.", room=room)


@socketio.on("create_instance")
def handle_test(data):
    user_name = data["userName"]
    camera_instance = Camera(mode="MTCNN")  # can choose HOG instead of MTCNN
    socketio.camera_instances_dict.setdefault(user_name, camera_instance)
    print(f"The camera_instances_dict is: {socketio.camera_instances_dict}")
    send(f"We create for user: {user_name} an instance of camera: {id(camera_instance)}", room=user_name)


@socketio.on("image")
def handle_image(image: str) -> None:
    # r = redis.Redis(connection_pool=redis_pool)
    # # we let "process_this_frame" decision happen in client-side
    # if image is not None:
    #     print(image)
    #     try:
    #         output_image_str = single_serve(image[Constant.JPEG_BASE64_STRING_HEADER_LENGTH.value::], r, mode="MTCNN")
    #     except Exception as e:
    #         emit("error", Exception)
    #     emit("result", output_image_str)
    session_id = flask.request.sid
    camera_instance: Camera = socketio.camera_instances_dict.get(session_id)
    username: str = camera_instance.serve_single_image(image)
    print(username)
    emit("result", username, room=session_id)


@socketio.on("connect")
def handle_connect():
    print(f"\nClient: {flask.request.sid} Connected")


@socketio.on("disconnect")
def handle_disconnect():
    camera_instance = socketio.camera_instances_dict.pop(flask.request.sid,
                                                         None)  # delete the key from the dictionary, this will return the camera_instance
    del camera_instance  # delete the camera instance immediatly, to improve the performance
    print(f"\nClient: {flask.request.sid} disconnected")
    print(f"The camera_instances_dict is: {socketio.camera_instances_dict}")


if __name__ == '__main__':
    # app.run(debug=True, threaded=True)
    # The flask_socketio encapsulates the start up of the web server
    socketio.run(app)
