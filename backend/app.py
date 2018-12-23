from flask import Flask, render_template, Response
import cv2
from flask_camera_service import Camera
from flask_socketio import SocketIO

app = Flask(__name__)
vc = cv2.VideoCapture(0)
socketio = SocketIO(app)


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


@socketio.on("image")
def handle_image(image: str) -> None:
    print(image)

@socketio.on("connect")
def handle_connect():
    print("\nClient Connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("\nClient disconnected")


if __name__ == '__main__':
    # app.run(debug=True, threaded=True)
    # The flask_socketio encapsulates the start up of the web server
    socketio.run(app)
