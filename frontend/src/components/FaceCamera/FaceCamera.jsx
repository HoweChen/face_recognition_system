import React, { Component } from 'react';
import Webcam from 'react-webcam';

export default class FaceCamera extends Component {
  setRef = (webcam) => {
    this.webcam = webcam;
  };

  componentDidMount() {
    this.connectToSocket();
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
      <Webcam audio={false}
        // height={480}
        // width={480}
        ref={this.setRef}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
        screenshotQuality={1}
      />
    );
  }
}
