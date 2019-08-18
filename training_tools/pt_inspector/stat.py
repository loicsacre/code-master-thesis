import numpy as np


class StreamingStat(object):
    """
    RunningStat
    ===========
    Computes the mean and std of some quantity given by batches of tensors
    """
    def __init__(self, size=0, first_mean=0, first_var=0,
                 running_mean=0, running_square_mean=0, running_var=0,
                 last_mean=0, last_var=0):
        self.size = size
        self.first_mean = first_mean
        self.first_var = first_var
        self.last_mean = last_mean
        self.last_var = last_var
        self.running_mean = running_mean
        self.running_square_mean = running_square_mean
        self.running_var = running_var  # Running within "addition" variance

    def add(self, np_tensor):
        # New values
        mean = np_tensor.mean()
        mean_sq = mean ** 2
        var = np_tensor.var()
        # Save last
        self.last_mean = mean
        self.last_var = var

        size = np.prod(np_tensor.shape)
        if self.size == 0:
            # First capture
            self.first_mean = self.last_mean
            self.first_var = self.last_var

        # Running stuff
        size_ratio_correction = self.size / float(self.size + size)
        size_ratio = size / float(self.size + size)
        self.size += size

        self.running_mean = mean * size_ratio \
            + self.running_mean * size_ratio_correction
        self.running_square_mean = mean_sq * size_ratio \
            + self.running_square_mean * size_ratio_correction
        self.running_var = var * size_ratio \
            + self.running_var * size_ratio_correction

    def get_running(self):
        btw_var = self.running_square_mean - self.running_mean**2
        wth_var = self.running_var
        return self.running_mean, np.sqrt(btw_var + wth_var)

    def get_first(self):
        return self.first_mean, np.sqrt(self.first_var)

    def get_last(self):
        return self.last_mean, np.sqrt(self.last_var)

    def reset(self):
        self.size = 0

    def __repr__(self):
        return "{cls}(size={size}, first_mean={first_mean}, " \
               "first_var={first_var}, running_mean={running_mean}, " \
               "running_square_mean={running_square_mean}, " \
               "running_var={running_var}, last_mean={last_mean}, " \
               "last_var={last_var})" \
               "".format(cls=self.__class__.__name__,
                         size=repr(self.size),
                         first_mean=repr(self.first_mean),
                         first_var=repr(self.first_var),
                         running_mean=repr(self.running_mean),
                         running_square_mean=repr(self.running_square_mean),
                         running_var=repr(self.running_var),
                         last_mean=repr(self.last_mean),
                         last_var=repr(self.last_var))

    def __str__(self):
        f_mean, f_std = self.get_first()
        l_mean, l_std = self.get_last()
        r_mean, r_std = self.get_running()
        return "First: {:.2E} +/- {:.2E} | Avg: {:.2E} +/- {:.2E} | " \
               "Last: {:.2E} +/- {:.2E}" \
               "".format(f_mean, f_std, r_mean, r_std, l_mean, l_std)
