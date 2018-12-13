import React, { Component } from 'react';
import CreativeRegister from './components/CreativeRegister';

export default class Register extends Component {
  static displayName = 'Register';

  constructor(props) {
    super(props);
    this.state = {};
  }

  render() {
    return (
      <div className="register-page">
        <CreativeRegister />
      </div>
    );
  }
}
