import bchlib
import glob
import os

import cv2
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from code_base.frame_size import FrameSize
from enum import Enum

BCH_POLYNOMIAL = 137
BCH_BITS = 5


class StegaStamp:
    # 这个累里传入的图像都是BGR原图，需要转成RGB然后缩放到400*400提取信息
    def __init__(self, mode="ENCODE", return_residual=False):
        self.return_residual = return_residual
        if mode == "ENCODE":
            print("Steagatamp instance is initializing...")
            MODEL_PATH = "/Users/howechen/GitHub/face_recognition_system/code_base/StegaStamp/saved_models/stegastamp_pretrained"
            self.encode_sess = tf.InteractiveSession(graph=tf.Graph())
            # model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)
            self.encode_model = tf.saved_model.loader.load(self.encode_sess, [tag_constants.SERVING], MODEL_PATH)

            input_secret_name = \
                self.encode_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
                    'secret'].name
            input_image_name = \
                self.encode_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
                    'image'].name
            self.encode_input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
            self.encode_input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

            output_stegastamp_name = \
                self.encode_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
                    'stegastamp'].name
            output_residual_name = \
                self.encode_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
                    'residual'].name
            self.encode_output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
            self.encode_output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)
        elif mode == "DECODE":
            pass

    def encode(self, input_raw_frames, asecret):
        width = FrameSize.WIDTH.value
        height = FrameSize.HEIGHT.value

        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

        if len(asecret) > 7:
            print('Error: Can only encode 56bits (7 characters) with ECC')
            return

        data = bytearray(asecret + ' ' * (7 - len(asecret)), 'utf-8')
        ecc = bch.encode(data)
        packet = data + ecc

        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0, 0, 0, 0])

        encoded_frames = []
        residual_frames = []

        # the input frame has already been set to RGB
        for image in input_raw_frames:
            # image = cv2.imread(frame)
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
            image = image.astype(np.float32)
            image /= 255.

            feed_dict = {self.encode_input_secret: [secret],
                         self.encode_input_image: [image]}

            hidden_img, residual = self.encode_sess.run([self.encode_output_stegastamp, self.encode_output_residual],
                                                        feed_dict=feed_dict)

            rescaled = (hidden_img[0] * 255).astype(np.uint8)
            encoded_frame = cv2.cvtColor(np.asarray(rescaled), cv2.COLOR_RGB2BGR)
            encoded_frames.append(encoded_frame)

            # get the residual image
            residual = residual[0] + .5
            residual = (residual * 255).astype(np.uint8)
            residual = cv2.cvtColor(np.squeeze(np.array(residual)), cv2.COLOR_BGR2RGB)
            residual_frames.append(residual)
        if self.return_residual is True:
            return encoded_frames, residual_frames
        else:
            return encoded_frames

    @staticmethod
    def decode(input_frame):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('model', type=str)
        parser.add_argument('--image', type=str, default=None)
        parser.add_argument('--images_dir', type=str, default=None)
        parser.add_argument('--secret_size', type=int, default=100)
        args = parser.parse_args()

        if args.image is not None:
            files_list = [args.image]
        elif args.images_dir is not None:
            files_list = glob.glob(args.images_dir + '/*')
        else:
            print('Missing input image')
            return

        sess = tf.InteractiveSession(graph=tf.Graph())

        model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

        input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
            'image'].name
        input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

        output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
            'decoded'].name
        output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

        bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

        for filename in files_list:
            image = Image.open(filename).convert("RGB")
            image = np.array(ImageOps.fit(image, (FrameSize.WIDTH, FrameSize.HEIGHT)), dtype=np.float32)
            image /= 255.

            feed_dict = {input_image: [image]}

            secret = sess.run([output_secret], feed_dict=feed_dict)[0][0]

            packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
            packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
            packet = bytearray(packet)

            data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

            bitflips = bch.decode_inplace(data, ecc)

            if bitflips != -1:
                try:
                    code = data.decode("utf-8")
                    print(filename, code)
                    continue
                except:
                    continue
            print(filename, 'Failed to decode')


if __name__ == '__main__':
    ss = StegaStamp()
    ss.encode(
        "/Users/howechen/GitHub/face_recognition_system/code_base/StegaStamp/images/in/Bureau-Collective-Diversity-of-the-Human-Face-04.jpg",
        [],
        "Hello")
