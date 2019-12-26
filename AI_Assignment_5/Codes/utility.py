def alphabetize(x,y):
    if x.get_name()>y.get_name():
        return 1
    return -1

def abs_mean(values):
    """Compute the mean of the absolute values a set of numbers.
    For computing the stopping condition for training neural nets"""
    return np.mean(np.abs(values))
