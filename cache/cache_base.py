import os.path
import pickle
import random
import string
import os


def generate_random_str(length=6):
    letters = string.ascii_lowercase
    random_str = ''.join(random.choice(letters) for i in range(10))
    return random_str


class CacheBase:
    def __init__(self, runner, cache_name=None, cache_dir="./tmp"):
        self.cache_path = os.path.join(cache_dir, "{}.pickle".format(cache_name))
        self.runner = runner
        self.cached_data = {}
        if os.path.exists(self.cache_path):
            fp = open(self.cache_path, "rb")
            self.cached_data = pickle.load(fp)
            fp.close()

    def get(self, query):
        return self.cached_data[query]

    def run(self, video_key, *inputs):
        if video_key in self.cached_data:
            return self.cached_data[video_key]
        else:
            print("new video key")
            result = self.runner(*inputs)
            self.cached_data[video_key] = result
            self.dump()
            return result

    def dump(self):
        fp = open(self.cache_path, "wb")
        pickle.dump(self.cached_data, fp)
        fp.close()


if __name__ == '__main__':
    generate_random_str()
