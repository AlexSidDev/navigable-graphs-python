import numpy as np
import argparse
from tqdm import tqdm
import time
# from tqdm import tqdm_notebook as tqdm
from heapq import heappush, heappop
import random
import itertools
import pickle

random.seed(108)
from hnsw_fast import HNSW
from hnsw import l2_distance, heuristic


def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def filter_verts(candidates, verts):
    filtered = []
    for candidate in candidates.keys():
        if candidate in verts:
            filtered.append(candidate)
    return filtered


def find_components_separate(graph: HNSW):
    levels = graph._graphs[::-1]
    components = [[] for _ in range(len(levels))]
    num_components = [0] * len(levels)

    for ind, level in enumerate(levels):
        verts = set(level.keys())
        while verts:
            components[ind].append([])
            candidates = [next(iter(verts))]
            while candidates:
                new_candidates = []
                verts.difference_update(candidates)
                for vert in candidates:
                    neighbours = level.get(vert, None)
                    if neighbours is not None:
                        new_candidates.extend(filter_verts(neighbours, verts))

                components[ind][-1].extend(candidates)
                candidates = new_candidates
            num_components[ind] += 1

    return num_components, components


def find_components(graph: HNSW):
    components = []
    num_components = 0

    levels = graph._graphs[::-1]
    verts = set(range(len(graph.data)))

    while verts:
        components.append([])
        candidates = [next(iter(verts))]
        while candidates:
            new_candidates = set()
            verts.difference_update(candidates)
            for vert in candidates:
                for level in levels:
                    neighbours = level.get(vert, None)
                    if neighbours is not None:
                        new_candidates.update(filter_verts(neighbours, verts))

            components[num_components].extend(candidates)
            candidates = list(new_candidates)

        num_components += 1

    return num_components, components


def main():
    parser = argparse.ArgumentParser(description='Finding graph components in HNSW')
    parser.add_argument('--dataset', default='base.10M.fbin',
                        help="Path to dataset file in .fbin format")
    parser.add_argument('--K', type=int, default=5, help='The size of the neighbourhood')
    parser.add_argument('--M', type=int, default=16, help='Avg number of neighbors')
    parser.add_argument('--M0', type=int, default=32, help='Avg number of neighbors')
    parser.add_argument('--dim', type=int, default=2, help='Dimensionality of synthetic data (ignored for SIFT).')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to search in the test stage')
    parser.add_argument('--ef', type=int, default=10, help='Size of the beam for beam search.')
    parser.add_argument('--ef_construction', type=int, default=64, help='Size of the beam for beam search.')
    parser.add_argument('--m', type=int, default=3, help='Number of random entry points.')

    args = parser.parse_args()

    vecs = read_fbin(args.dataset)

    # Create HNSW

    hnsw = HNSW(distance_type='l2', m=args.M, m0=args.M0, ef=args.ef_construction)

    # Add data to HNSW
    for x in tqdm(vecs):
        hnsw.add(x)

    start = time.time()
    num_components, components = find_components_separate(hnsw)
    print('Количество компонент для каждого уровня:', num_components)

    print('Время выполнения:', time.time() - start, 'с')

    start = time.time()
    num_components, components = find_components(hnsw)
    print('Количество компонент в \"плоском\" графе:', num_components)

    print('Время выполнения:', time.time() - start, 'с')


if __name__ == "__main__":
    main()
