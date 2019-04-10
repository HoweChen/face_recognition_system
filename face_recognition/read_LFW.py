import pathlib
import redis

folders = pathlib.Path("/Users/howechen/Dropbox/Lab/dataset/CV/lfw")


if __name__ == '__main__':
    redis_pool = redis.ConnectionPool()
    r = redis.Redis(connection_pool=redis_pool)


    for sub_folder in folders.iterdir():
        if sub_folder.is_dir():
            folders.open()