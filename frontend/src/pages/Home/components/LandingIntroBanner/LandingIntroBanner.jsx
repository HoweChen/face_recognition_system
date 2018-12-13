import React, { Component } from "react";
import { Button } from "@icedesign/base";
import "./LandingIntroBanner.scss";

export default class LandingIntroBanner extends Component {
  static displayName = "LandingIntroBanner";

  static propTypes = {};

  static defaultProps = {};

  constructor(props) {
    super(props);
    this.state = {};
  }

  render() {
    return (
      <div className="landing-intro-banner" style={{ height: "100vh" }}>
        <div
          className="landing-intro-banner-background"
          style={{
            backgroundImage: `url(${require("./images/pedro-lastra-157071-unsplash.jpg")})`,
            backgroundPosition: "center",
          }}
        />
        <div className="landing-intro-banner-content-wrapper">
          <div className="landing-intro-banner-content">
            <h2 style={styles.title}>Face Recognition System</h2>
            <p style={styles.subTitle}>
              Support user face detection and recognition.
            </p>
            <div
              className="landing-intro-banner-buttons"
              style={{ textAlign: "center", marginTop: 70 }}
            >
              <a href="//alibaba.github.io/ice/block" style={styles.leftButton}>
                <Button
                  style={{
                    height: 50,
                    padding: "0 58px",
                    fontSize: 16,
                    marginBottom: "20px",
                    color: "#3080FE",
                  }}
                  size="large"
                  type="normal"
                >
                  Login
                </Button>
              </a>
              <a href="//localhost:/Signup">
                <Button
                  style={{
                    height: 50,
                    padding: "0 58px",
                    fontSize: 16,
                    marginBottom: "20px",
                  }}
                  type="primary"
                  size="large"
                >
                  Register
                </Button>
              </a>
            </div>

            <div style={{ marginTop: "80px", position: "relative" }}>
              <div style={styles.gitContainer}>
                <iframe
                  style={styles.gitStar}
                  title="HoweChen/face-recognition-system"
                  src="https://ghbtns.com/github-btn.html?user=HoweChen&repo=face_recognition_system&type=star&count=true&size=large"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

const styles = {
  title: {
    textAlign: "center",
    fontSize: "70px",
    letterSpacing: "4px",
    lineHeight: "60px",
    color: "#fff",
    marginBottom: "30px",
  },
  subTitle: {
    fontSize: "30px",
    margin: "0px",
    color: "#fff",
    textShadow: "#C8C8C8 1px 1px 2px",
    textAlign: "center",
    lineHeight: "1.7em",
  },
  leftButton: {
    marginRight: "20px",
  },
  gitStar: {
    border: "0px",
    height: "32px",
    width: "145px",
    margin: "0 auto",
  },
  gitContainer: {
    marginTop: "30px",
    textAlign: "center",
  },
  updateLogLinkWrap: {
    textAlign: "center",
  },
  updateLogLink: {
    color: "#fff",
  },
};
