# License: MIT

import multiprocessing.pool

class ParallelEvaluation(object):
    def __init__(self, objective_function, n_worker=1):
        self.n_worker = n_worker
        self.process_pool = None
        self.objective_function = objective_function

    def parallel_execute(self, param_list, callback=None):
        results = list()
        apply_results = list()

        for _param in param_list:
            apply_results.append(self.process_pool.apply_async(
                self.objective_function,
                (_param,), callback=callback)
            )
        for res in apply_results:
            res.wait()
            perf = res.get()
            results.append(perf)
        return results

    def __enter__(self):
        self.process_pool = ProcessPool(processes=self.n_worker)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process_pool.close()



class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# Sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class ProcessPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(ProcessPool, self).__init__(*args, **kwargs)