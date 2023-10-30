
import multiprocessing as mp
import time

PROCESSES = 4
cacca =  0
class TEST:

    def __init__(self, todo_queue, results_queue, barrier):
        self.todo_queue = todo_queue
        self.results_queue = results_queue
        self.barrier = barrier

    def run_worker(self, lock):
        
        if not self.todo_queue.empty():
            caches = []
            interval = self.todo_queue.get()
            for i in interval:
                caches.append(self.work(i, lock))
            self.barrier.wait()
            self.results_queue.put(self.local_work(caches))

    def work(self, i, lock):
        cache = []
        cache.append(i)
        lock.acquire()
        cacca = len(cache)
        lock.release()
        return cache
    
    def local_work(self, caches):
        cand = []
        for c in caches:
            cand.append(max(c))
        return cand
