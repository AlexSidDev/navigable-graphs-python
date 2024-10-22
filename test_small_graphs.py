import time
# from tqdm import tqdm_notebook as tqdm
import random

random.seed(108)
from hnsw_fast import HNSW


def filter_verts(candidates, verts):
    filtered = []
    for candidate in candidates.keys():
        if candidate in verts:
            filtered.append(candidate)
    return filtered


def find_components_separate(graph: HNSW):
    levels = graph  # hardcode для тестового графа
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

    levels = graph
    verts = set(range(1, 7)) # hardcode для тестового графа

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

    graph = [{1: {2: 0, 3: 0}, 2: {1: 0, 3: 0}, 3: {1: 0, 2: 0},
               4: {5: 0, 6: 0}, 5: {4: 0, 6: 0}, 6: {4: 0, 5: 0}},
               {1: {5: 0}, 5: {1: 0}}] # вершины в списке соседей представлены в виде {вершина: дистанция} для совместимости с функцией поиска

    start = time.time()
    num_components, components = find_components_separate(graph)
    print('Количество компонент для каждого уровня:', num_components)
    print('Компоненты:', components)

    print('Время выполнения:', time.time() - start, 'с')

    start = time.time()
    num_components, components = find_components(graph)
    print('Количество компонент в \"плоском\" графе:', num_components)
    print('Компоненты:', components)

    print('Время выполнения:', time.time() - start, 'с')


if __name__ == "__main__":
    main()
