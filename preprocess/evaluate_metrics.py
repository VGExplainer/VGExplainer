def dice_coefficient(a, b):
    a_bigrams = set(a)
    b_bigrams = set(b)
    overlap = len(a_bigrams & b_bigrams)
    return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))

def calculateIou(a, b):
    interarea = list(set(a).intersection(set(b)))
    union = list(set(a).union(b))
    return float(len(interarea) / len(b))

def calculatePrecison(a, b):
    interarea = list(set(a).intersection(set(b)))
    return float(len(interarea) / len(b))

def calculateAccuracy(a, b):
    a_bigrams = set(a)
    b_bigrams = set(b)
    overlap = len(a_bigrams & b_bigrams)
    return overlap
