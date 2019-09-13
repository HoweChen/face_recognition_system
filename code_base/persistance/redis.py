import redis


class Redis:
    def __init__(self):
        self.redis_pool = redis.ConnectionPool()
        self.r = None
        self.connect()

    def connect(self):
        try:
            self.r = redis.Redis(connection_pool=self.redis_pool)
        except ConnectionError as e:
            print(e)
        finally:
            return self


redis_client = Redis()
