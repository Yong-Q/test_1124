from experiments.xi_positive_total_pair.operators import StableTotalPair

CHANNELS = (1, 2, 13, 14, 16)


def make_model():
    return StableTotalPair(channels=CHANNELS)
