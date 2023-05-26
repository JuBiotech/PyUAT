""" Module for simple sampling functionality"""

import numbers

import numpy as np


class Probability:
    """Probability value that uses log probability for computations"""

    def __init__(self, probability=None, log_prob=None):
        assert probability is None or log_prob is None

        if probability is not None:
            assert np.all(probability > 0)
            self.logProb = np.log(probability)
        else:
            self.logProb = log_prob

    @property
    def probability(self):
        return np.exp(self.logProb)

    @property
    def log_probability(self):
        return self.logProb

    def __eq__(self, other):
        if isinstance(other, numbers.Number):
            return self.probability == other
        elif isinstance(other, Probability):
            return self.logProb == other.logProb
        else:
            # unsupported equals operator
            assert False

    def __pow__(self, other):
        assert isinstance(other, numbers.Number)
        return Probability(log_prob=self.logProb * other)

    def __mul__(self, other):
        temp = other
        if not isinstance(other, Probability):
            temp = Probability(other)
        return Probability(log_prob=self.logProb + temp.logProb)

    def __rmul__(self, other):
        """
        other: some kind of number (int, float) that is used on left side
        """
        return self.__mul__(Probability(other))

    def __truediv__(self, other):
        temp = other
        if not isinstance(other, Probability):
            temp = Probability(other)
        return Probability(log_prob=self.logProb - temp.logProb)

    def __rtruediv__(self, other):
        temp = other
        if not isinstance(other, Probability):
            temp = Probability(other)
        return Probability(log_prob=temp.logProb - self.logProb)

    def __add__(self, other):
        # According to https://en.wikipedia.org/wiki/Log_probability
        # log(x + y) = x' + log(1 + exp(y' - x')) where x'=log(x) and y'=log(y)
        temp = other
        if not isinstance(other, Probability):
            temp = Probability(other)
        return Probability(
            log_prob=self.logProb + np.log1p(np.exp(temp.logProb - self.logProb))
        )

    def __radd__(self, other):
        if other == 0:
            # Special case: if we add 0 to probability it stays unchanged!
            return Probability(log_prob=self.logProb)
        return self.__add__(Probability(other))

    def __sub__(self, other):
        # According to https://en.wikipedia.org/wiki/Log_probability
        # log(x + y) = x' + log(1 + exp(y' - x')) where x'=log(x) and y'=log(y)
        # therefore log(x - y) = x' + log(1 - exp(y' - x'))
        temp = other
        if not isinstance(other, Probability):
            temp = Probability(other)

        assert np.all(self.log_probability >= temp.log_probability)

        return Probability(
            log_prob=self.logProb + np.log1p(-np.exp(temp.logProb - self.logProb))
        )

    def __rsub__(self, other):
        if isinstance(other, float):
            assert other > 0 and np.all(np.log(other) >= self.log_probability)

            other = Probability(other)

        return other.__sub__(self)

    def __str__(self):
        return f"Probability(value: {self.probability:.4f}, log: {self.logProb:.2f})"


def log_prob_dist_to_prob_dist(raw_log_probabilities):
    """
    Takes a list of log probablities, normalizes them and converts them to probability values

    raw_log_probabilities: list of log probabilities
    """
    # convert to our probability space
    probabilites = [
        Probability(log_prob=log_prob) for log_prob in raw_log_probabilities
    ]
    # compute discrete distribution normalizer
    normalizer = sum(probabilites)

    return [(prob / normalizer).probability for prob in probabilites]


def sample_from_log_probabilities(raw_log_probabilities, num_samples):
    """
    Samples indices from a raw log probability list
    """
    prob_dist = log_prob_dist_to_prob_dist(raw_log_probabilities)

    return np.random.choice(
        range(len(raw_log_probabilities)), size=num_samples, p=prob_dist
    )


def normalize_prob_dist(probabilities):
    normalizer = sum(probabilities)
    return [prob / normalizer for prob in probabilities]


def sample_from_probabilities(probabilities, num_samples):
    normed_dist = normalize_prob_dist(probabilities)
    value_probabilities = [prob.probability for prob in normed_dist]

    return np.random.choice(
        range(len(probabilities)), size=num_samples, p=value_probabilities
    )


if __name__ == "__main__":
    # small test
    p = Probability(np.linspace(0.1, 0.7, 7))
    p2 = Probability(np.linspace(0.2, 0.8, 7, dtype=np.float32))

    print(p.log_probability.shape, p2.log_probability.shape)

    newProb = 1.0 - (p2 - p)

    print(newProb.probability)
