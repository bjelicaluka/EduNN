from random import shuffle


def shuffle_data(inputs, labels):
    list_for_shuffle = list(zip(inputs, labels))
    shuffle(list_for_shuffle)
    return zip(*list_for_shuffle)


def count_decimals(n):
    k = 0
    while n % 1 != 0:
        n *= 10
        k += 1
    return k