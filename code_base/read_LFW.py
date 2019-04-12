import pathlib
import redis
from code_base.camera_service import Instance

folders = pathlib.Path("/Users/howechen/Dropbox/Lab/dataset/CV/lfw")

if __name__ == '__main__':
    ins = Instance(mode="MTCNN", f_e_m="NORMAL")

    redis_pool = redis.ConnectionPool()
    r = redis.Redis(connection_pool=redis_pool)
    count = 0
    failed_count = 0
    for sub_folder in folders.iterdir():
        if sub_folder.is_dir():
            sub_folder = pathlib.Path(sub_folder)
            for file in sub_folder.iterdir():
                if file.is_file():
                    count += 1
                    try:
                        file_name = file.name.replace(".jpg", "")
                        ins.image_to_db(filepath=str(file.absolute()), name=str(file_name), image_size="SMALL",
                                        num_jitters=10)
                    except Exception as e:
                        print(e)
                        failed_count += 1
    print(f"total: {count}, failed_count: {failed_count}")
