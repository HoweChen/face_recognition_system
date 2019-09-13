import pathlib
import redis
import time
from code_base.camera_service_dev import Instance
import multiprocessing.dummy as T
import multiprocessing as P
from code_base.experiments.register_to_db import register_to_db
import numpy as np


def register_dataset(ins, path):
    # register lfw5590
    count = 0
    failed_count = 0
    thread = T.Pool(processes=None)
    folders = pathlib.Path(path)
    for file in folders.iterdir():
        if file.is_file():
            count += 1
            if count > 5590:
                break
            else:
                try:
                    file_name = file.name.replace(".jpg", "")
                    thread.apply(register_to_db, args=(ins, file, file_name,))
                    # register_to_db(ins, file, file_name)
                except Exception as e:
                    print(e)
                    failed_count += 1
                print(f"Processed: {count}")
    thread.close()
    thread.join()
    thread.terminate()
    print(f"total: {count - 1}, failed_count: {failed_count}")


if __name__ == '__main__':
    redis_pool = redis.ConnectionPool()
    r = redis.Redis(connection_pool=redis_pool)
    # epoches = [10, 100, 500, 1000, 2500, 5000, 5590]
    # epoches = [10, 100, 500, 1000, 2500, 5000, 10000, -2]  # use -2 because -2+1 = -1
    watermark = True
    # dataset_path = ["/Users/howechen/Dropbox/Lab/dataset/CV/MTFL/lfw_5590",
    #                 # "/Users/howechen/Dropbox/Lab/dataset/CV/MTFL/AFLW",
    #                 # "/Users/howechen/Dropbox/Lab/dataset/CV/MTFL/net_7876"
    #                 ]
    tolerance_list = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    with open("./result_ARF1.txt", mode="w") as f_open:
        for _ in range(1):
            ins = Instance(mode="MTCNN", f_e_m="NORMAL", watermark=watermark)
            # for dataset in dataset_path:
            #     register_dataset(ins, dataset)
            # register myself to db

            test_dataset_path = pathlib.Path("/Users/howechen/Dropbox/Lab/dataset/CV/lfw_multiple_images")
            for tolerance in tolerance_list:
                TP = 0
                FP = 0
                FN = 0
                count = 0
                for test_image in test_dataset_path.iterdir():
                    if test_image.is_file() is True:
                        count += 1
                        # print(test_image.name)
                        test_name = test_image.name
                        test_name = test_name[0:-9]
                        # print(test_name)
                        try:
                            return_face_encoding, return_name = ins.match_single_face(str(test_image.absolute()),
                                                                                      image_size="SMALL",
                                                                                      num_jitters=1,
                                                                                      tolerance=tolerance)
                        except Exception as e:
                            FN += 1
                        if return_name == "Unknown":
                            FN += 1
                        else:
                            return_name = return_name[0:-5]
                            if test_name == return_name:
                                TP += 1
                            else:
                                FP += 1
                        print(f"file count: {count},{test_name},{return_name}")
                A = TP / (TP + FP)
                R = TP / (TP + FN)
                F1 = 2 * A * R / (A + R)
                print(f"TP: {TP}")
                print(f"FP: {FP}")
                print(f"FN: {FN}")
                print(f"A: {A}")
                print(f"R: {R}")
                print(f"F1 = {F1}")
                f_open.write(f"tolerance: {tolerance}\n")
                f_open.write(f"TP: {TP}\n")
                f_open.write(f"FP: {FP}\n")
                f_open.write(f"FN: {FN}\n")
                f_open.write(f"A: {A}\n")
                f_open.write(f"R: {R}\n")
                f_open.write(f"F1: {F1}\n")
                f_open.write("------------\n")

            # f_open.write(f"with_watermark: {watermark}\n")
            # for epoch in epoches:
            #     times = []
            #     for i in range(10):
            #         start = float(time.time())
            #         name = ins.match_single_face(me_test_image_path, image_size="SMALL", num_jitters=1,
            #                                      match_range=epoch + 1)
            #         end = float(time.time())
            #         times.append(end - start)
            #         print(end - start)
            #     # print(times)
            #     times_str = "\n".join(map(str, times))
            #     f_open.write(f"\nEpoch: {epoch}\n{times_str}")
            # # f_open.write("\n------------------\n")
            # # watermark = not watermark
        f_open.close()
