import React, { Component } from 'react';
import CreativeLogin from './components/CreativeLogin';

export default class Login extends Component {
  static displayName = 'Login';

  constructor(props) {
    super(props);
    this.state = {};
  }

  render() {
    return (
      <div className="login-page">
        <CreativeLogin />
      </div>
    );
  }
}
