from __future__ import division
from __future__ import print_function

from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from operator import itemgetter
import numpy as np
import sys
import time
import heapq
from .unionfind import UnionFind
# import time
# import multiprocessing
# from random import random
from multiprocessing import shared_memory, current_process
# from functools import partial
# import timeit
# from line_profiler import LineProfiler
# import cProfile, pstats, io
# from pstats import SortKey
try:
    from math import log2
except ImportError:  # Python 2.x or <= 3.2
    from math import log

    def log2(x):
        return log(x, 2)


inf = float("inf")
MISSING = sys.maxsize
MISSING_WEIGHT = sys.float_info.max


class HNSW:
    def __init__(
        self,
        distance,
        data,
        members,
        levels,
        positions,
        shm_adj,
        shm_weights,
        shm_hnsw_data,
        shm_enter_point,
        shm_count,
        lock,
        m=5,
        ef=32,
        m0=None,
    ):
        self.data = data
        self.dim = len(data)
        self.distance = distance
        self.min_samples = 5
        self.distance_cache = {}
        self.shm_enter_point = shm_enter_point
        self.shm_count = shm_count
        self.sh_point = np.ndarray(shape=(1), dtype=int, buffer=shm_enter_point.buf)
        self.hnsw_data = np.ndarray(
            shape=(self.dim), dtype=int, buffer=shm_hnsw_data.buf
        )
        self.sh_count = np.ndarray(shape=(1), dtype=int, buffer=shm_count.buf)

        self.members = members
        self.levels = levels
        self.positions = positions

        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._enter_point = None

        # the hnsw graph now is composed by two numpy array for each level
        # that contain one the adjacency list of each elem,
        # and the other the weights list of each edges
        self.shm_adj = shm_adj
        self.shm_weights = shm_weights

        self.shared_weights = []
        self.shared_adjs = []
        for i in range(len(self.members)):
            self.shared_adjs.append(
                np.ndarray(
                    shape=(len(self.members[i]), self._m0 if i == 0 else self._m),
                    dtype=int,
                    buffer=shm_adj[i].buf,
                )
            )
            self.shared_weights.append(
                np.ndarray(
                    shape=(len(self.members[i]), self._m0 if i == 0 else self._m),
                    dtype=float,
                    buffer=shm_weights[i].buf,
                )
            )
        self.lock = lock

    def decorated_d(self, distance_cache, i, j):
        if j in distance_cache:
            return distance_cache[j]
        distance_cache[j] = dist = self.distance(self.data[i], self.data[j])
        return dist

    def add_and_compute_local_mst(self, points):
        distances = []
        for point in points:
            distances.append(self.hnsw_add(point))
        time_localMST = 0
        start = time.time()
        local_mst = self.local_mst(distances, points)
        end = time.time()
        time_localMST = end - start
        return local_mst, time_localMST

    def hnsw_add(self, elem):
        distance_cache = {}
        level = self.calc_level(elem)
        """Add elem to the data structure"""
        sh_point = (
            np.ndarray(shape=(1), dtype=int, buffer=self.shm_enter_point.buf) + MISSING
            if elem == 0
            else np.ndarray(shape=(1), dtype=int, buffer=self.shm_enter_point.buf)
        )
        sh_count = np.ndarray(shape=(1), dtype=int, buffer=self.shm_count.buf)

        enter_point = sh_point[0]
        hnsw_data = self.hnsw_data
        hnsw_data[elem] = elem

        idx = elem
        ef = self._ef
        d = self.distance
        m = self._m

        shared_weights = []
        shared_adjs = []

        if enter_point != MISSING:  # the HNSW is not empty, we have an entry point
            dist = self.decorated_d(distance_cache, elem, enter_point)
            sh_count[0] = sh_count[0] + 1
            level_sh_point = self.calc_level(enter_point)

            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            if level_sh_point > level:
                level_to_search_pos = level_sh_point - 1
                for i in range(len(self.members)):
                    shared_adjs.append(
                        np.ndarray(
                            shape=(len(self.members[i]), self._m0 if i == 0 else m),
                            dtype=int,
                            buffer=self.shm_adj[i].buf,
                        )
                    )
                    shared_weights.append(
                        np.ndarray(
                            shape=(len(self.members[i]), self._m0 if i == 0 else m),
                            dtype=float,
                            buffer=self.shm_weights[i].buf,
                        )
                    )
                for g1, g2, sh1, sh2 in zip(
                    reversed(shared_adjs[level:level_sh_point]),
                    reversed(shared_weights[level:level_sh_point]),
                    reversed(self.shm_adj[level:level_sh_point]),
                    reversed(self.shm_weights[level:level_sh_point]),
                ):
                    level_m = self._m0 if level_to_search_pos == 0 else m
                    enter_point, dist = self._search_graph_ef1(
                        sh_count,
                        level_to_search_pos,
                        idx,
                        enter_point,
                        dist,
                        g1,
                        sh1,
                        distance_cache,
                        level_m,
                    )
                    level_to_search_pos = level_to_search_pos - 1
            # at these levels we have to insert elem; ep is a heap of
            # entry points.
            ep = [(-dist, enter_point)]
            level_mod = level_sh_point if level_sh_point < level else level
            level_to_search_pos = level_mod - 1

            for i in range(len(self.members)):
                shared_adjs.append(
                    np.ndarray(
                        shape=(len(self.members[i]), self._m0 if i == 0 else m),
                        dtype=int,
                        buffer=self.shm_adj[i].buf,
                    )
                )
                shared_weights.append(
                    np.ndarray(
                        shape=(len(self.members[i]), self._m0 if i == 0 else m),
                        dtype=float,
                        buffer=self.shm_weights[i].buf,
                    )
                )

            for g1, g2, sh1, sh2 in zip(
                reversed(shared_adjs[:level_mod]),
                reversed(shared_weights[:level_mod]),
                reversed(self.shm_adj[:level_mod]),
                reversed(self.shm_weights[:level_mod]),
            ):
                level_m = self._m0 if level_to_search_pos == 0 else m
                ep = self._search_graph(
                    sh_count,
                    level_to_search_pos,
                    idx,
                    ep,
                    g1,
                    sh1,
                    distance_cache,
                    level_m,
                    ef,
                )

                pos = self.positions[level_to_search_pos].get(idx)
                self._select_heuristic(
                    level_to_search_pos,
                    pos,
                    idx,
                    ep,
                    level_m,
                    g1,
                    g2,
                    sh1,
                    sh2,
                    heap=True,
                )
                # insert backlinks to the new node
                for j, dist in zip(g1[pos], g2[pos]):
                    if j == MISSING or dist == MISSING_WEIGHT:
                        break
                    pos2 = self.positions[level_to_search_pos].get(j)
                    self._select_heuristic(
                        level_to_search_pos,
                        pos2,
                        j,
                        (idx, dist),
                        level_m,
                        g1,
                        g2,
                        sh1,
                        sh2,
                    )

                level_to_search_pos = level_to_search_pos - 1

        if enter_point == MISSING or self.calc_level(enter_point) < level:
            self.lock.acquire()
            if enter_point == MISSING or self.calc_level(enter_point) < level:
                sh_point[0] = elem
            self.lock.release()

        return distance_cache

    def printResult(self, shm_adj, shm_weights, members):
        adjs = []
        weights = []
        for shm1, shm2, memb, i in zip(
            shm_adj, shm_weights, members, range(len(members))
        ):
            adj = np.ndarray(
                shape=(len(memb), self._m0 if i == 0 else self._m),
                dtype=int,
                buffer=shm1.buf,
            )
            adjs.append(adj)
            weight = np.ndarray(
                shape=(len(memb), self._m0 if i == 0 else self._m),
                dtype=float,
                buffer=shm2.buf,
            )
            weights.append(weight)
        print("weights: ", weights, "\n")
        print("Adjacency: ", adjs, "\n")

    def calc_position(self, to_find, level_to_search):
        # pos = 0
        # for i, p in zip(self.members[level_to_search], range(len(self.members[level_to_search]))):
        #     # print(j)
        #     if i == to_find:
        #         return p
        return self.positions[level_to_search].get(to_find)

    def calc_level(self, elem):
        for dic, i in zip(
            reversed(self.positions), reversed(range(len(self.positions)))
        ):
            if elem in dic:
                return i + 1

    def search(self, graphs, q, k=None, ef=None):
        """Find the k points closest to q."""

        d = self.distance
        graphs = graphs
        sh_point = np.ndarray(shape=(1), dtype=int, buffer=self.shm_enter_point.buf)
        # sh_point = self.sh_point
        point = sh_point[0]
        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        dist = d(q, self.data[point])
        # look for the closest neighbor from the top to the 2nd level
        for g in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1_test(q, point, dist, g)
        # look for ef neighbors in the bottom level
        ep = self._search_graph_test(q, [(-dist, point)], graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1_test(self, q, entry, dist, g):
        """Equivalent to _search_graph when ef=1."""

        d = self.distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])
        while candidates:
            # print(candidates, "\n\n", visited)
            dist, c = heappop(candidates)
            # print("q ef1:", q)
            # print("g[c]",g[c])
            if dist > best_dist:
                break
            edges = [e for e in g[c] if e not in visited]
            # print("edges: ",edges)
            # print("g[c]: ", g[c])
            if not edges:
                continue
            visited.update(edges)
            dists = [d(q, data[e]) for e in edges]
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))
                    # break

        return best, best_dist

    def _search_graph_test(self, q, ep, g, ef):
        d = self.distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break

            edges = [e for e in g[c] if e not in visited]
            if not edges:
                continue
            # print("edges 2: ",edges)
            visited.update(edges)
            dists = [d(q, data[e]) for e in edges]
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _search_graph_ef1(
        self,
        count_dist,
        level_to_search,
        q,
        entry,
        dist,
        arr_adj,
        shm_adj,
        distance_cache,
        m,
    ):
        g_adj = np.ndarray(shape=arr_adj.shape, dtype=int, buffer=shm_adj.buf)
        d = self.distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            pos = self.calc_position(c, level_to_search)
            edges = []
            for e in g_adj[pos]:
                if e == MISSING:
                    break
                if e not in visited:
                    edges.append(e)

            if not edges:
                continue
            visited.update(edges)
            count_dist[0] = count_dist[0] + len(edges)
            # dists = [d(data[q], data[e]) for e in edges]
            dists = [self.decorated_d(distance_cache, q, e) for e in edges]

            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))
        return best, best_dist

    def _search_graph(
        self,
        count_dist,
        level_to_search,
        q,
        ep,
        arr_adj,
        shm_adj,
        distance_cache,
        m,
        ef,
    ):
        g_adj = np.ndarray(shape=(len(arr_adj), m), dtype=int, buffer=shm_adj.buf)

        d = self.distance
        data = self.data
        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)
        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:  # ??
                break
            pos = self.calc_position(c, level_to_search)
            edges = []
            for e in g_adj[pos]:
                if e == MISSING:
                    break
                if e not in visited:
                    edges.append(e)
            if not edges:
                continue

            visited.update(edges)
            count_dist[0] = count_dist[0] + len(edges)
            # dists = [d(data[q], data[e]) for e in edges]
            dists = [self.decorated_d(distance_cache, q, e) for e in edges]

            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _select_naive(self, d, to_insert, m, g, heap=False):
        if not heap:  # shortcut when we've got only one thing to insert
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        # so we have more than one item to insert, it's a bit more tricky
        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)  # smallest m distances
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            assert len(d) == m

    def _select_heuristic(
        self,
        level_to_search,
        position,
        elem,
        to_insert,
        m,
        arr_adj,
        arr_weights,
        shm_adj,
        shm_weights,
        heap=False,
    ):
        g_adj = np.ndarray(shape=arr_adj.shape, dtype=int, buffer=shm_adj.buf)
        g_weights = np.ndarray(
            shape=arr_weights.shape, dtype=float, buffer=shm_weights.buf
        )

        def prioritize(idx, dist):
            b = False
            for ndw, nda in zip(nb_dicts_weights, nb_dicts_adj):
                p = np.where(nda == idx)[0]
                if len(p) == 0:
                    if inf < dist:
                        b = True
                        break
                elif ndw[p[0]] < dist:
                    b = True
                    break
            return b, dist, idx

        nb_dicts_adj = []
        nb_dicts_weights = []
        for idx, i in zip(g_adj[position], range(len(g_adj[position]))):
            if idx == MISSING:
                break
            pos = self.calc_position(idx, level_to_search)
            nb_dicts_adj.append(g_adj[pos])
            nb_dicts_weights.append(g_weights[pos])

        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
            if idx in g_adj[position]:
                to_insert = []
        else:
            tempList1 = []
            for mdist, idx in to_insert:
                if idx in g_adj[position]:
                    continue
                tempList1.append(prioritize(idx, -mdist))
            to_insert = nsmallest(m, tempList1)

        # assert len(to_insert) > 0
        assert not any(
            idx in g_adj[position] for _, _, idx in to_insert
        ), "idx:{0}".format(
            idx
        )  # check if the assert make sense in concurrent version
        ll = list(filter(lambda i: i != MISSING, g_adj[position]))
        unchecked = m - len(ll)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)

        if to_check > 0:
            tempList2 = []
            for idx, dist in zip(g_adj[position], g_weights[position]):
                if idx == MISSING:
                    break
                tempList2.append(prioritize(idx, dist))
            checked_del = nlargest(to_check, tempList2)

        else:
            checked_del = []
        # with self.elem_locks[lock_id]:
        for _, dist, idx in to_insert:
            for i, el in enumerate(g_weights[position]):
                if el == MISSING_WEIGHT:
                    g_weights[position][i] = abs(dist)
                    g_adj[position][i] = idx
                    break

        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            # with self.elem_locks[lock_id]:
            for i, el in enumerate(g_adj[position]):
                if el == idx_old:
                    # pos = i
                    g_adj[position][i] = idx_new
                    g_weights[position][i] = abs(d_new)
                    break
        # assert list(filter(lambda i: i != MISSING, d_adj)) == m and list(filter(lambda i: i != MISSING_WEIGHT, d_weights)) == m]

    def local_mst(self, distances, points):
        shared_weights = []
        shared_adjs = []
        for i in range(len(self.members)):
            shared_adjs.append(
                np.ndarray(
                    shape=(len(self.members[i]), self._m0 if i == 0 else self._m),
                    dtype=int,
                    buffer=self.shm_adj[i].buf,
                )
            )
            shared_weights.append(
                np.ndarray(
                    shape=(len(self.members[i]), self._m0 if i == 0 else self._m),
                    dtype=float,
                    buffer=self.shm_weights[i].buf,
                )
            )
        data = self.data
        candidate_edges = []
        points = list(points)
        for d_cache, i in zip(distances, points):
            if i in d_cache:
                d_cache.pop(i)
            nhi = shared_weights[0][i]
            nhi = np.sort(nhi)
            for j, dist in d_cache.items():
                assert dist == self.distance(self.data[i], self.data[j])
                nhj = shared_weights[0][j]
                nhj = np.sort(nhj)
                candidate_edges.append(
                    (max(dist, nhi[self._m], nhj[self._m]), i, j, dist)
                )
        mst_edges = []
        n = len(data)
        uf = UnionFind(n)
        heapify(candidate_edges)
        while candidate_edges:
            mrd, i, j, dist = heappop(candidate_edges)
            if uf.union(i, j):
                mst_edges.append((mrd, i, j, dist))
        return mst_edges

    def global_mst(self, shm_adj, shm_weights, candidate_edges, n):
        # edges = []
        # for  mrd, i, j, dist in candidate_edges:
        #     assert dist == self.distance(self.data[i], self.data[j])
        #     nhi = self.shared_weights[0][i]
        #     nhj = self.shared_weights[0][j]
        #     nhi = np.sort(nhi)
        #     nhj = np.sort(nhj)
        #     edges.append((max(nhi[self._m], nhj[self._m], dist),i,j, dist ))
        # heapify(edges)
        # heapify(candidate_edges)
        needed_edges = len(self.data) - 1
        uf = UnionFind(n)
        final_mst = []
        # while needed_edges:
        #     _, i, j, _ = node =heappop(candidate_edges)
        #     if uf.union(i, j):
        #         final_mst.append((node))
        #         needed_edges -= 1
        # return final_mst
        candidate_edges.sort()
        for mrd, i, j, dist in candidate_edges:
            if uf.union(i, j):
                final_mst.append((mrd, i, j, dist))
        return final_mst

    # def global_mst(self, distances_cache):

        # candidate_edges = []
        # nh = []
        # new_edges = {}
        # data = self.data
        # min_samples = self.min_samples
        # minus_infty = -np.infty
        # total_count = 0
        # total_sum = 0

        # for _ in range(len(data)):
        #     heap = [(minus_infty, minus_infty)] * min_samples
        #     heapq.heapify(heap)
        #     nh.append(heap)

        # for idx, dist_cache in zip(range(len(data)), distances_cache):
        #     if idx in dist_cache:
        #         dist_cache.pop(idx)
        #     # print(dist_cache, len(dist_cache), idx)
        #     total_count += len(dist_cache)
        #     # dist_cache = dict(sorted(dist_cache.items(), key=lambda item: item[0]))
        #     for j, dist in dist_cache.items():
        #         # print(dist, idx, j)
        #         total_sum += dist
        #         mdist = -dist
        #         heapq.heappushpop(nh[idx], (mdist, j))
        #         new_edges[j, idx] = dist

        #         # also update j's reachability distances
        #         nh_j = nh[j]
        #         old_mrd = heapq.heappushpop(nh_j, (mdist, idx))[0]
        #         new_mrd = nh_j[0][0]
        #         if old_mrd != new_mrd:
        #             # i is a new close neighbor for j and j's reachability
        #             # distance changed
        #             for md, k in nh_j:
        #                 if k == idx or k == minus_infty:
        #                     continue
        #                 if nh[k][0][0] > old_mrd:
        #                     # reachability distance between j and k decreased
        #                     key = (j, k) if j < k else (k, j)
        #                     new_edges[key] = -min(md, new_mrd)
        #     # print("nh", nh, "\n")
        #     # print("new edges",new_edges, "\n")
        #     # print(idx, " -- distance cache: ",dist_cache, "\n")
        # # print("new edges par: ", new_edges)
        # candidate_edges.extend(
        #     (max(dist, -nh[i][0][0], -nh[j][0][0]), i, j, dist)
        #     for (i, j), dist in new_edges.items()
        # )
        # heapq.heapify(candidate_edges)
        # # print("parall cand edges: ", candidate_edges)
        # # Kruskal's algorithm

        # mst_edges = []
        # n = len(data)
        # needed_edges = n - 1
        # uf = UnionFind(n)
        # while needed_edges:
        #     _, i, j, _ = edge = heapq.heappop(candidate_edges)
        #     if uf.union(i, j):
        #         mst_edges.append(edge)
        #         needed_edges -= 1
        # return mst_edges

    # def _select_heuristic(self, elem_locks, level_to_search, position, elem, to_insert, m, g_adj, g_weights, heap=False):
    #     def priority(idx, dist):
    #         triang_dists = neighbor_weights[neighbor_adj == idx]
    #         return np.any(triang_dists < dist)

    #     lock_id = elem % MAX_LABEL_OPERATION_LOCKS
    #     # print("lock id: ", lock_id)

    #     pos_adj = g_adj[position]
    #     assert m == pos_adj.shape[0]
    #     pos_missing = np.where(pos_adj == MISSING)[0]
    #     n_neighbors = pos_missing[0] if pos_missing.size else pos_adj.size
    #     assert pos_missing.size + n_neighbors == m
    #     assert 0 <= n_neighbors <= m
    #     pos_adj = pos_adj[:n_neighbors]
    #     assert not np.any(pos_adj == MISSING)

    #     # neighbor_positions = [self.calc_position(idx, level_to_search) for idx in pos_adj]
    #     level_positions = self.positions[level_to_search]
    #     neighbor_positions = [level_positions[idx] for idx in pos_adj]
    #     neighbor_adj, neighbor_weights = g_adj[neighbor_positions], g_weights[neighbor_positions]
    #     if not heap:
    #         idx, dist = to_insert
    #         if idx in pos_adj:
    #             return

    #         if n_neighbors < m:
    #             g_adj[position, n_neighbors] = idx
    #             g_weights[position, n_neighbors] = dist
    #         else:
    #             pri = priority(idx, dist)
    #             old_pri, old_dist, old_pos = max((priority(idx, dist), dist, i)
    #                                             for i, (idx, dist) in enumerate(zip(pos_adj, g_weights[position])))
    #             if pri < old_pri or (pri == old_pri and dist < old_dist):
    #                 g_adj[position, old_pos] = idx
    #                 g_weights[position, old_pos] = dist
    #     else:
    #         # we generally get here with n_neighbors = 0
    #         insert_queue = []
    #         for mdist, idx in to_insert:
    #             assert mdist <= 0
    #             if idx in pos_adj:
    #                 continue
    #             dist = -mdist
    #             # in general len(to_insert) > m, so it's not important to optimize away the case where we don't need priority
    #             insert_queue.append((priority(idx, dist), dist, idx))
    #         queue_length = len(insert_queue)

    #         assert not any(idx in pos_adj for _, _, idx in insert_queue), f"idx:{idx}" #check if the assert make sense in concurrent version
    #         assert m == g_adj[position].shape[0]
    #         unchecked = min(m - n_neighbors, queue_length)
    #         assert 0 <= unchecked <= m
    #         to_check = queue_length - unchecked
    #         check_slots = m - unchecked
    #         checking = to_check and check_slots  # this is generally false because check_slots = 0
    #         insert_queue = nsmallest(m, insert_queue)
    #         if checking:
    #             insert_queue, checked_ins = insert_queue[:unchecked], insert_queue[unchecked:]

    #         #with elem_locks[lock_id]:
    #         _, distances, indices = zip(*insert_queue)  # zip(*[[1, 2, 3], [4, 5, 6]]) = [(1, 4), (2, 5), (3, 6)]
    #         right_end = n_neighbors + unchecked
    #         g_adj[position, n_neighbors:right_end] = indices
    #         g_weights[position, n_neighbors:right_end] = distances

    #         if checking:  # we take this branch very seldom
    #             # print("to check:", to_check)
    #             checked_del = ((priority(idx, dist), dist, i)
    #                         for i, (idx, dist) in enumerate(zip(pos_adj, g_weights[position])))
    #             checked_del = nlargest(check_slots, checked_del)
    #             for (p_new, d_new, idx_new), (p_old, d_old, old_pos) in zip(checked_ins, checked_del):
    #                 if (p_old, d_old) <= (p_new, d_new):
    #                     break
    #                 g_adj[position, old_pos] = idx_new
    #                 g_weights[position, old_pos] = d_new

    # def __getitem__(self, idx):
    #     """Returns a list of known neighbors of node at index idx."""

    #     for g in self._graphs:
    #         try:
    #             yield from g[idx].items()
    #         except KeyError:
    #             return
