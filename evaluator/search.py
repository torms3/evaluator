from __future__ import print_function

from collections import OrderedDict
import numpy as np

import segascorus as sc

from .metric.voi import voi
from .metric.rand import adapted_rand


class LineSearch(object):
    def __init__(self, depth=1, start=0.1, stop=1.0, step=0.1):
        self.depth = depth
        self.start = start
        self.stop = stop
        self.step = step

    def __call__(self, seg1, seg2, metric, cmp_func=np.argmin):
        self.seg1 = seg1
        self.seg2 = seg2

        self.results = OrderedDict()
        self.opt_thresh = 0.0

        # Initial grid search
        grid = np.arange(self.start, self.stop, self.step)
        step = self.step
        self.grid_search(grid, metric)

        # Iteration
        for d in range(self.depth):
            # Coarse search
            self.find_optimum(metric, cmp_func)
            step /= 2.0
            grid = np.array([-1,1])
            grid = self.opt_thresh + (step * grid)
            self.grid_search(grid, metric)

            # Fine search
            self.find_optimum(metric, cmp_func)
            step /= 5.0
            grid = np.array([-4,-3,-2,-1,1,2,3,4])
            grid = self.opt_thresh + (step * grid)
            self.grid_search(grid, metric)

        # Find optimal threshold.
        self.find_optimum(metric, cmp_func)

    def find_optimum(self, metric, cmp_func):
        idx = cmp_func([v[metric]["error"] for v in self.results.values()])
        self.opt_thresh = list(self.results.items())[idx][0]
        print("Optimal threshold = {}".format(self.opt_thresh))

    def grid_search(self, grid, metric):
        for t in grid:
            if t in self.results:
                continue
            print("Threshold = {:.3f}".format(t))
            seg1 = self.seg1(threshold=t)
            seg2 = self.seg2(threshold=t)
            # self.results[t] = sc.score(seg1, seg2)
            if metric == "voi":
                split, merge = voi(seg1, seg2)
                error = merge + split
            elif metric == "rand":
                error, merge, split = adapted_rand(seg1, seg2, all_stats=True)
            else:
                assert False
            self.results[t][metric]["error"] = error
            self.results[t][metric]["merge"] = merge
            self.results[t][metric]["split"] = split
            print("{} merge: {:.3f}".format(metric, merge))
            print("{} split: {:.3f}".format(metric, split))
            print("{} error: {:.3f}".format(metric, error))
            print("")

    def save(self, fpath):
        pass
