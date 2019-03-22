/* eslint react/no-string-refs:0 */
import React, { Component } from 'react';
import { Feedback } from '@icedesign/base';
import AuthForm from './AuthForm';

export default class LoginFrom extends Component {
  static displayName = 'LoginFrom';

  static propTypes = {};

  static defaultProps = {};

  formChange = (value) => {
    console.log('formChange:', value);
  };

  handleSubmit = (errors, values) => {
    if (errors) {
      console.log('errors', errors);
      return;
    }
    console.log('values:', values);
    Feedback.toast.success('Login Success!');
    // 登录成功后做对应的逻辑处理
  };

  render() {
    const config = [
      {
        label: 'Username',
        component: 'Input',
        componentProps: {
          placeholder: 'Username',
          size: 'large',
          maxLength: 20,
        },
        formBinderProps: {
          name: 'name',
          required: true,
          message: 'This field is required',
        },
      },
      {
        label: 'Password',
        component: 'Input',
        componentProps: {
          placeholder: 'Password',
          htmlType: 'passwd',
        },
        formBinderProps: {
          name: 'passwd',
          required: true,
          message: 'This field is required',
        },
      },
      {
        label: 'Remember username',
        component: 'Checkbox',
        componentProps: {},
        formBinderProps: {
          name: 'checkbox',
        },
      },
      {
        label: 'Face Recognition',
        component: 'Switch',
        componentProps: {},
        formBinderProps: {
          name: 'switch',
        },
      },
      {
        label: 'Login',
        component: 'Button',
        componentProps: {
          type: 'primary',
        },
        formBinderProps: {},
      },
    ];

    const initFields = {
      name: '',
      passwd: '',
      checkbox: false,
    };

    const links = [
      { to: '/register', text: 'Register now!' },
      { to: '/forgetpassword', text: 'Forgot the password?' },
    ];

    return (
      <AuthForm
        title="Login"
        config={config}
        initFields={initFields}
        formChange={this.formChange}
        handleSubmit={this.handleSubmit}
        links={links}
        changeFaceMode={this.props.changeFaceMode}
        isFaceModeOn={this.props.isFaceModeOn}
        username={this.props.username}
      />
    );
  }
}
