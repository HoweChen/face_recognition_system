import cv2
import pathlib
import os


def jpg_compression(img):
    rates = [1, 5, 10, 25, 50, 75, 90, 100]
    compressed_images = []
    for rate in rates:
        compression = [int(cv2.IMWRITE_JPEG_QUALITY), rate]
        # cv2.imwrite(f"/Users/howechen/GitHub/face_recognition_system/code_base/scaffold/result{rate}.jpg", img, compression)
        rtv, encoded_image = cv2.imencode(".jpg", img, compression)
        compressed_image = cv2.imdecode(encoded_image, 1)
        compressed_images.append(compressed_image)
    return compressed_images


def median_filter(img):
    rates = [1, 3, 5, 7, 9]
    compressed_images = []
    for rate in rates:
        compressed_image = cv2.medianBlur(img, rate)
        compressed_images.append(compressed_image)
    return compressed_images


def gaussian_filter(path: str):
    rates = [1, 3, 5, 7, 9]
    compressed_images = []
    for rate in rates:
        compressed_image = cv2.GaussianBlur(img, (rate, rate), 0)
        compressed_images.append(compressed_image)
    return compressed_images


if __name__ == '__main__':
    # raw_path = "/Users/howechen/Dropbox/Lab/dataset/CV/lfw/Robert_Downey_Jr/Robert_Downey_Jr_0001.jpg"
    # img = cv2.imread(raw_path)
    # # rates = [1, 5, 10, 25, 50, 75, 90, 100]
    # rates = [1, 3, 5, 7, 9]
    # for rate in rates:
    #     # compression = [int(cv2.IMWRITE_JPEG_QUALITY), rate]
    #     compressed_image = cv2.GaussianBlur(img, (rate, rate), 0)
    #     # cv2.imwrite(f"/Users/howechen/GitHub/face_recognition_system/code_base/scaffold/result{rate}.jpg", img, compression)
    #     # rtv, encoded_image = cv2.imencode(".jpg", img, compression)
    #     # print(rtv)
    #     # cv2.imshow(f"result{rate}", cv2.imdecode(encoded_image, 1))
    #     cv2.imshow(f"result{rate}", compressed_image)
    #     cv2.imwrite(
    #         f"/Users/howechen/GitHub/face_recognition_system/code_base/scaffold/gaussian_filter{rate}x{rate}.jpg",
    #         compressed_image)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     exit(1)
    raw_path_dir = pathlib.Path("/Users/howechen/Dropbox/Lab/dataset/CV/lfw_multiple_images")
    count = 0
    funcs = [jpg_compression, median_filter, gaussian_filter]
    for file in raw_path_dir.iterdir():
        print(file.name)
        count += 1
        if count >= 1001:
            break
        if file.is_file():
            img = cv2.imread(str(file.absolute()))
            for func in funcs:
                print(func.__name__)
                result = func(img)
                for i in range(len(result)):
                    compressed_image = result[i]
                    result_name = file.name.replace(".jpg", "") + f"_{func.__name__}" + str(i) + ".jpg"
                    cv2.imwrite(f"/Users/howechen/Dropbox/Lab/dataset/CV/compressed_images/{result_name}",
                                compressed_image)
