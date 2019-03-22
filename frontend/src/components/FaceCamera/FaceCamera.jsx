import React, { Component } from 'react';
import Webcam from 'react-webcam';
import { Button } from '@icedesign/base';

export default class FaceCamera extends Component {
  setRef = (webcam) => {
    this.webcam = webcam;
  };

  constructor(props) {
    super(props);
    this.capture = this.capture.bind(this);
  }

  componentDidMount() {
    this.connectToSocket();
  }

  componentWillUnmount() {
    if (this.socket !== null) {
      this.disconnect();
    }
  }

  connectToSocket = () => {
    const io = require('socket.io-client');
    const socket = io('http://127.0.0.1:5000', {
      'sync disconnect on unload': true,
    });
    this.setState({
      socket,
    });

    socket.on('message', (message) => {
      console.log('message :', message);
    });

    socket.on('connect', () => {
      socket
        .emit('join', {
          userName: socket.id.toString(),
          room: socket.id.toString(),
        })
        .emit('create_instance', {
          userName: socket.id.toString(),
        });
    });
    socket.on('disconnect', () => {
      // this handles the situation when the server is shutdown
      console.log('Disconnect from the server');
      socket.close();
    });

    socket.on('result', (username) => {
      console.log(username);
      this.setState({
        username,
      });
      this.props.setUsername(username);
    });

    socket.on('error', (error) => {
      console.log(error);
    });
  };

  disconnect = () => {
    this.state.socket.close();
  };

  capture = () => {
    const imageSrc = this.webcam.getScreenshot();
    this.state.socket.emit('image', imageSrc);
  };

  render() {
    const videoConstraints = {
      width: 480,
      height: 480,
      facingMode: 'user',
    };
    return (
      <div style={styles.container}>
        <Webcam audio={false}
        // height={480}
        // width={480}
                ref={this.setRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                screenshotQuality={1}
        />
        <Button
          style={styles.button}
          onClick={this.capture}
        >
          Capture Image
        </Button>
      </div>
    );
  }
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    height: '100vh',
  },
  logoLink: {
    display: 'block',
  },
  logoImg: {
    width: '88px',
  },
  title: {
    marginTop: '60px',
    fontWeight: '500',
    fontSize: '22px',
    lineHeight: '1.5',
    textAlign: 'center',
    color: '#343a40',
  },
  description: {
    marginTop: '30px',
    fontSize: '13px',
    color: '#212529',
  },
  button: {
    marginTop: '40px',
    width: '180px',
    height: '48px',
    lineHeight: '48px',
    textAlign: 'center',
    borderRadius: '50px',
    border: '1px solid #9816f4',
  },
  border: {
    position: 'absolute',
    top: '100px',
    bottom: '100px',
    right: '0',
    background: '#ffffff',
    width: '30px',
    boxShadow: '-19px 0 35px -7px #F5F5F5',
  },
};
