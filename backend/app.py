from flask import Flask, render_template, Response
import cv2
from flask_camera_service import Camera
from flask_socketio import SocketIO, join_room, leave_room, send
import redis

app = Flask(__name__)
vc = cv2.VideoCapture(0)
socketio = SocketIO(app)
redis_pool = redis.ConnectionPool()


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
    print(socketio.room_users)


@socketio.on("leave")
def on_leave(data):
    user_name = data["userName"]
    room = data["room"]
    leave_room(room, sid=room)
    send(f"{user_name} has left the room: {room}.", room=room)


@socketio.on("create_instance")
def handle_test(data):
    user_name = data["userName"]
    room = data["userName"]
    camera_instance = Camera(mode="MTCNN")
    send(f"We create for user: {user_name} an instance of camera: {id(camera_instance)}", room=room)


@socketio.on("image")
def handle_image(image: str) -> None:
    # print(image)
    # print(id(camera_instance))
    r = redis.Redis(connection_pool=redis_pool)



@socketio.on("connect")
def handle_connect():
    print("\nClient Connected")
    # camera_instance = Camera(mode="MTCNN")
    # print(camera_instance)


@socketio.on("disconnect")
def handle_disconnect():
    print("\nClient disconnected")


if __name__ == '__main__':
    # app.run(debug=True, threaded=True)
    # The flask_socketio encapsulates the start up of the web server
    socketio.run(app)
