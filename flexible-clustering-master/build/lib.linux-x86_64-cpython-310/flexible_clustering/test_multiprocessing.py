import multiprocessing as mp
from itertools import pairwise
import sklearn.datasets 
import class_test
import time
import importlib
importlib.reload(class_test)
PROCESSES = 16
def split(a, n):
    k, m = divmod(len(a), n)
    indices = [k * i + min(i, m) for i in range(n + 1)]
    return [a[l:r] for l, r in pairwise(indices)]

if __name__ == '__main__':  
    mp.set_start_method('fork')

    data, labels = sklearn.datasets.make_blobs(20000, centers=5, random_state=10)
    intervals = split(range(1, len(data)), PROCESSES)
    
    todo_queue = mp.Queue()
    results_queue = mp.Queue()
    barrier = mp.Barrier(PROCESSES)
    lock = mp.Lock()

    for interval in intervals:
        todo_queue.put(interval)

    test = class_test.TEST(todo_queue, results_queue, barrier)
    processes =  []
    for interval in intervals:
        # print(interval)
        process = mp.Process(target=test.run_worker, args=(lock,))
        processes.append(process)
        process.start()

    # processes = [mp.Process(target=test.run_worker) for _ in range(PROCESSES)]
    candidates = []

    for process in processes:
        process.join()

    while not results_queue.empty():
        candidates.extend(results_queue.get())
    
    print(candidates)

    # locks = [manager.Lock() for _ in range(6)]
    # todo_queue = multiprocessing.Queue(maxsize=num_processes)
    # results_queue = multiprocessing.Queue(maxsize=num_processes)
    
    # barrier = multiprocessing.Barrier(num_processes)
    # hnsw = hnsw_parallel.HNSW(calc_dist, data, members, levels, positions,
    #                           shm_adj, shm_weights, shm_hnsw_data, shm_ent_point, shm_count, todo_queue, results_queue, barrier, lock, locks, m=m, m0=m0, ef=32)

    # intervals = split(range(1, len(data)), num_processes)
    # for interval in intervals:
    #     todo_queue.put(interval)
    
    # processes =  []
    # for i in range(num_processes):
    #     # print(interval)
    #     process = multiprocessing.Process(target=hnsw.add_and_compute_local_mst)
    #     print(process)
    #     process.daemon = True
    #     process.start()
    #     processes.append(process)
        
    # for process in processes:
    #     process.join()
    #     print(process)
    # # processes = [multiprocessing.Process(target=hnsw.add_and_compute_local_mst) for _ in range(num_processes)]
    
    # while not results_queue.empty():
    #     candidate_edges.extend(results_queue.get())

    # todo_queue.close()
    # results_queue.close()

        # def add_and_compute_local_mst(self):
    #     # print(f"{process_name} is working")
    #     if not self.todo_queue.empty():
    #         interval = self.todo_queue.get()
    #         caches = []
    #         for point in interval:
    #             caches.append(self.hnsw_add(point))
    #         # print(f"{process_name} has reached the specific point")
    #         self.barrier.wait()
    #         # print(f"{process_name} continues after the barrier 1")
    #         local_mst = self.local_mst(caches, interval)
    #         self.results_queue.put(local_mst)