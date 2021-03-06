import pathlib
import redis
from code_base.camera_service import Instance
from multiprocessing import Pool

folders = pathlib.Path("/Users/howechen/Dropbox/Lab/dataset/CV/lfw")


def register_to_db(file, file_name):
    ins = Instance()
    try:
        ins.image_to_db(filepath=str(file.absolute()), name=str(file_name), image_size="SMALL",
                             num_jitters=1)
    except Exception as e:
        raise e


if __name__ == '__main__':
    # ins = Instance(mode="MTCNN", f_e_m="NORMAL")
    # TODO: 实现多进程，现在只是主进程把文件放进多进程里，但是没有用
    redis_pool = redis.ConnectionPool()
    r = redis.Redis(connection_pool=redis_pool)
    count = 0
    failed_count = 0
    p = Pool(processes=None)
    for sub_folder in folders.iterdir():
        if sub_folder.is_dir():
            sub_folder = pathlib.Path(sub_folder)
            for file in sub_folder.iterdir():
                if file.is_file():
                    count += 1
                    try:
                        file_name = file.name.replace(".jpg", "")
                        # ins.image_to_db(filepath=str(file.absolute()), name=str(file_name), image_size="SMALL",
                        #                 num_jitters=1)
                        p.apply(register_to_db, args=(file, file_name,))
                    except Exception as e:
                        print(e)
                        failed_count += 1
                    print(f"Processed: {count}")
    print(f"total: {count}, failed_count: {failed_count}")
