from __future__ import print_function

from collections import OrderedDict
import numpy as np

import segascorus as sc


class LineSearch(object):
    def __init__(depth=1, start=0.1, stop=1.0, step=0.1):
        self.depth = depth
        self.start = start
        self.stop = stop
        self.step = step

    def __call__(seg1, seg2, metric, cmp_func):
        self.seg1 = seg1
        self.seg2 = seg2

        self.results = OrderedDict()
        self.opt_thresh = 0.0

        # Initial grid search
        grid = np.arange(self.start, self.stop, self.step)
        step = self.step
        self.grid_search(grid)

        # Iteration
        for d in range(self.depth):
            # Coarse search
            self.find_optimum(metric, cmp_func)
            step /= 2.0
            grid = [-1,1]
            grid = self.opt_thresh + (step * grid)
            self.grid_search(grid)

            # Fine search
            self.find_optimum(metric, cmp_func)
            step /= 5.0
            grid = [-4,-3,-2,-1,1,2,3,4]
            grid = self.opt_thresh + (step * grid)
            self.grid_search(grid)

        # Find optimal threshold.
        self.find_optimum(metric, cmp_func)

    def find_optimum(self, metric, cmp_func):
        idx = cmp_func([v[metric] for v in self.results.values()])
        self.opt_thresh = list(results.items())[idx][0]
        print("Optimal threshold = {}".format(self.opt_thresh))

    def grid_search(self, grid):
        for t in grid:
            if t in self.results:
                continue
            print("Threshold = {:.3f}".format(t))
            self.results[t] = sc.score(self.seg1(t), self.seg2(t))
            print("")
