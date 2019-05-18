import pathlib
import redis
import time
from code_base.camera_service_dev import Instance
import multiprocessing.dummy as T
import multiprocessing as P
from code_base.experiments.register_to_db import register_to_db


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
    epoches = [10, 100, 500, 1000, 2500, 5000, 10000, -2]  # use -2 because -2+1 = -1
    r.flushall()
    watermark = False
    dataset_path = ["/Users/howechen/Dropbox/Lab/dataset/CV/MTFL/lfw_5590",
                    # "/Users/howechen/Dropbox/Lab/dataset/CV/MTFL/AFLW",
                    # "/Users/howechen/Dropbox/Lab/dataset/CV/MTFL/net_7876"
                    ]
    test_path = "/Users/howechen/Downloads/lfw_multiple_images"

    with open("./result.txt", mode="w") as f_open:
        for _ in range(1):
            r.flushall()
            ins = Instance(mode="MTCNN", f_e_m="NORMAL", watermark=watermark)
            for dataset in dataset_path:
                register_dataset(ins, dataset)
            # register myself to db
            me_image_path = "/Users/howechen/GitHub/face_recognition_system/code_base/debug/me.png"
            me_test_image_path = "/Users/howechen/GitHub/face_recognition_system/code_base/debug/me_test.jpg"
            my_name = "Howe Chen"
            register_to_db(ins, me_image_path, my_name)
            f_open.write(f"with_watermark: {watermark}\n")
            for epoch in epoches:
                times = []
                for i in range(10):
                    start = float(time.time())
                    name = ins.match_single_face(me_test_image_path, image_size="SMALL", num_jitters=1,
                                                 match_range=epoch + 1)
                    end = float(time.time())
                    times.append(end - start)
                    print(end - start)
                # print(times)
                times_str = "\n".join(map(str, times))
                f_open.write(f"\nEpoch: {epoch}\n{times_str}")
            f_open.write("\n------------------\n")
            watermark = not watermark
        f_open.close()
