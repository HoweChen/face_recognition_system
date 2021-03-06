import React, { Component } from 'react';
import { Grid } from '@icedesign/base';
import LoginIntro from './LoginIntro';
import LoginForm from './LoginForm';

const { Row, Col } = Grid;

export default class CreativeLogin extends Component {
  static propTypes = {};

  static defaultProps = {};

  constructor(props) {
    super(props);
    this.state = {
      isFaceModeOn: false,
      username: null,
    };
    this.changeFaceMode = this.changeFaceMode.bind();
  }

  changeFaceMode = () => {
    this.setState(prevState => ({
      isFaceModeOn: !prevState.isFaceModeOn,
    }));
  };

  setUserName=(username) => {
    this.setState({
      username,
    });
  }


  render() {
    return (
      <div style={styles.container}>
        <Row wrap>
          <Col l="12">
            <LoginIntro isFaceModeOn={this.state.isFaceModeOn} setUsername={username => this.setUsername(username)} />
          </Col>
          <Col l="12">
            <div style={styles.content}>
              <LoginForm changeFaceMode={() => this.changeFaceMode} isFaceModeOn={this.state.isFaceModeOn} username={this.state.username} />
            </div>
          </Col>
        </Row>
      </div>
    );
  }
}

const styles = {
  container: {
    position: 'relative',
    width: '100wh',
    minWidth: '1200px',
    height: '100vh',
    backgroundImage: `url(${require('./images/bg.jpg')})`,
  },
  content: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
  },
};
