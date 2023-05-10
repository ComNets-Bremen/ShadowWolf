
# Create batches of iterables
def batch(iterable, n=4):
    l = len(iterable)
    for ndx in range(0,l,n):
        yield iterable[ndx:min(ndx + n, l)]
