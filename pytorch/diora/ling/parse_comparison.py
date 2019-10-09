import os
import json
import heapq
from itertools import chain
import nltk


def nltk_tree_to_tuples(tree):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return tr[0]
        if len(tr) == 1:
            return helper(tr[0])
        return tuple([helper(x) for x in tr])
    return helper(tree)


def tuples_to_spans(tree):
    """
    Returns list of spans, that are (start, size).
    """

    result = []

    def helper(tr, pos=0):
        if isinstance(tr, str):
            return 1
        size = 0
        for x in tr:
            subsize = helper(x, pos=pos+size)
            size += subsize
        result.append((pos, size))
        return size

    helper(tree)

    return result

class Tree(object):
    def __init__(self):
        pass

    @classmethod
    def build_from_berkeley(cls, ex):
        result = cls()
        result.tree = nltk_tree_to_tuples(nltk.Tree.fromstring(ex['tree'].strip())) #19983
        result.spans = tuples_to_spans(result.tree)
        result.example_id = ex.get('example_id', ex.get('exampled_id', None))
        assert result.example_id is not None, "No example id for Berkeley parse."
        return result

    @classmethod
    def build_from_diora(cls, ex):
        result = cls()
        result.tree = ex['tree']
        result.spans = tuples_to_spans(result.tree)
        result.example_id = ex['example_id']
        assert result.example_id is not None, "No example id for Diora parse."
        return result



def read_supervised(path):
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            try:
                yield Tree.build_from_berkeley(ex)
            except:
                print('Skipping {}'.format(ex))

def read_unsupervised(path):
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            try:
                yield Tree.build_from_diora(ex)
            except:
                print('Skipping {}'.format(ex))

def find_intersection(sup, unsup):
    both = dict()
    for id in unsup:
        if id in sup.keys():
            spans = dict()
            spans['supervised'] = sup[id].spans
            spans['unsupervised'] = unsup[id].spans
            both[id] = spans
    return both

def calculate_statistics(intersect, n=10):
    stats = dict()
    stats['scores'] = dict()
    hi = []
    lo = []
    avg_pre = 0
    avg_rec = 0
    counter = 0
    for id in intersect:
        stats['scores'][id] = dict()
        truth = intersect[id]['supervised']
        pred = intersect[id]['unsupervised']
        
        if len(truth) == 0 or len(pred) == 0:
            continue
        # calculate precision (how many predictions were relevant) 
        common = set(truth).intersection(set(pred))
        correct = [x for x in common]
        print(correct, truth, pred)
        correct_predictions = len(correct)
        stats['scores'][id]['precision'] = correct_predictions / len(pred)
        avg_pre += correct_predictions / len(pred)

        # calculate recall (how many relevant items were selected)
        stats['scores'][id]['recall'] = correct_predictions / len(truth)
        avg_rec += correct_predictions / len(truth)

        # calculate f1 score
        pre = stats['scores'][id]['precision']
        rec = stats['scores'][id]['recall']
        f1 = float("-inf")
        if pre + rec != 0:
            f1 = 2 * (pre * rec) / (pre + rec)
        stats['scores'][id]['f1'] = f1

        # update hi
        max_data = (f1, id)
        min_data = (-f1, id)
        heapq.heappush(hi, max_data)
        if len(hi) > n:
            heapq.heappop(hi)
        
        # update lo
        heapq.heappush(lo, min_data)
        if len(lo) > n:
            heapq.heappop(lo)
        
        counter += 1

    converted_lo = []
    for i in range(len(lo)):
        f1, id = lo[i]
        converted_lo.append((-f1, id))

    stats['highest'] = hi
    stats['lowest'] = converted_lo
    stats['average_precision'] = avg_pre / counter
    stats['average_recall'] = avg_rec / counter

    return stats

def main(options):
    cache = {}

    # Read the supervised.
    cache['supervised'] = {}
    for x in read_supervised(options.supervised):
        cache['supervised'][x.example_id] = x
    
    # Read the unsupervised.
    cache['unsupervised'] = {}
    for x in read_unsupervised(options.unsupervised):
        cache['unsupervised'][x.example_id] = x

    print("supervised size:", len(cache['supervised']), "unsupervised size:", len(cache['unsupervised']))
    
    # Measure the closeness of unsupervised to supervised. 
    
    # find the keys that intersect, store in dict
    intersect = find_intersection(cache['supervised'], cache['unsupervised'])
    print(len(intersect))
    
    # Compute the recall per sentence, and average.
    stats = calculate_statistics(intersect)

    print(stats)
    with open("nv_dev.stats", "a") as f:
        f.write(json.dumps(stats))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervised', default=None, type=str)
    parser.add_argument('--unsupervised', default=None, type=str)
    options = parser.parse_args()
    main(options)
