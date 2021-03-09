from .utils import xp, parameterize, product


@xp
@parameterize(['size'], product({
    'size': [100, 2 ** 29],
}))
class TimeBasicManipulation(object):
    def time_arange(self, xp, size):
        xp.arange(size, dtype=xp.float32)
