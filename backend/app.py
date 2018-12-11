from flask import Flask, Response, render_template

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/camera')
def camera():
    return


if __name__ == '__main__':
    app.run()
