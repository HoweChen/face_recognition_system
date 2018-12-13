from typing import Generator
from multiprocessing import Queue


def count(n: int) -> Generator[int, None, None]:
    result = 0
    while n < 10:
        yield result + n
        n += 1


if __name__ == '__main__':
    for num in subprocess.run(count(9), stdout=subprocess.PIPE):
        print(num)
