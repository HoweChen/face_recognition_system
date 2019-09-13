import pathlib
import redis
import time
from code_base.camera_service_dev import Instance
import multiprocessing.dummy as T
import multiprocessing as P
from code_base.experiments.register_to_db import register_to_db


def register_dataset(ins, path):
    # register yale-b
    count = 0
    failed_count = 0
    # thread = T.Pool(processes=None)
    folders = pathlib.Path(path)
    for file in folders.iterdir():
        if file.is_file():
            count += 1
            # if count > 10:
            #     break
            file_name = file.name[0:-12]
            print(file_name)
            try:
                # thread.apply(register_to_db, args=(ins, file, file_name,))

                ins.yaleb_to_db(str(file.absolute()), file_name, num_jitters=1)
                # register_to_db(ins, file, file_name)
            except Exception as e:
                print(e)
                failed_count += 1
            # print(file.name)
            print(f"Processed: {count}")
    # thread.close()
    # thread.join()
    # thread.terminate()
    print(f"total: {count}, failed_count: {failed_count}")


if __name__ == '__main__':
    # ins = Instance(mode="MTCNN", f_e_m="NORMAL", watermark=False)
    redis_pool = redis.ConnectionPool()
    r = redis.Redis(connection_pool=redis_pool)
    # r.flushall()

    dataset_path = [
        "/Users/howechen/Dropbox/Lab/dataset/CV/classified_yale_b"
    ]
    test_dataset_list = [
        "/Users/howechen/Dropbox/Lab/dataset/CV/compressed_images/gaussian",
        "/Users/howechen/Dropbox/Lab/dataset/CV/compressed_images/jpg_compression",
        "/Users/howechen/Dropbox/Lab/dataset/CV/compressed_images/median",
    ]

    with open("./compressed_images_test.txt", mode="w") as f_open:
        # r.flushall()
        watermark = False
        ins = Instance(mode="MTCNN", f_e_m="NORMAL", watermark=watermark)
        # for dataset in dataset_path:
        #     register_dataset(ins, dataset)
        #     r.save()
        for test_dataset in test_dataset_list:
            test_dataset_path = pathlib.Path(test_dataset)
            f_open.write("\n------------------\n")
            f_open.write(f"{test_dataset_path.name}\n")
            for herd in test_dataset_path.iterdir():
                if herd.is_dir():
                    f_open.write(f"{herd.name}\n")
                    count = 0
                    true_count = 0
                    false_count = 0
                    for file in herd.iterdir():
                        if file.name == ".DS_Store":
                            continue
                        count += 1
                        print(file.absolute())
                        return_result = ins.detect_image_only(filepath=str(file.absolute()), image_size="SMALL")
                        print(return_result)
                        if return_result:
                            true_count += 1
                        else:
                            false_count += 1
                    f_open.write(f"Count: {count}\n")
                    f_open.write(f"True: {true_count}\n")
                    f_open.write(f"False: {false_count}\n")
                    f_open.write("\n------------------\n")
        f_open.close()

# f_open.write(f"with_watermark: {watermark}\n")
# for epoch in epoches:
#     times = []
#     for i in range(10):
#         print(i)
#         start = float(time.time())
#         face_encoding, name = ins.match_single_face(me_test_image_path, image_size="SMALL", num_jitters=1,
#                                                     match_range=epoch + 1)
#         end = float(time.time())
#         times.append(end - start)
#         print(end - start)
#     # print(times)
#     times_str = "\n".join(map(str, times))
#     f_open.write(f"\nEpoch: {epoch}\n{times_str}")
