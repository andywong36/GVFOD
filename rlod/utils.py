
class IncrMeanVar:
    def __init__(self):
        self.n = 0
        self.mu = None
        self.S = None
        self.var = None

    def incr(self, x):
        # increase n
        self.n += 1

        # update the mean
        if self.n == 1:
            self.mu = x
            last_mu = 0
        else:
            last_mu = self.mu
            self.mu = self.mu + (x - self.mu) / (self.n)

        # update the variance
        if self.n == 1:
            self.S = 0
        else:
            self.S = self.S + (x - last_mu) * (x - self.mu)
        self.var = self.S / self.n

    def get(self):
        return self.mu, self.var


