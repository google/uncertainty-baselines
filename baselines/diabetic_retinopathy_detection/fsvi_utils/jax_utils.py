import jax


class KeyHelper:
    def __init__(self, key):
        self._key = key

    def next_key(self):
        self._key, sub_key = jax.random.split(self._key)
        return sub_key
