from __future__ import print_function

import cloudvolume as cv
from cloudvolume.lib import Vec, Bbox


def cutout(opt, dtype='uint8'):
    print(opt.gs_input)

    # CloudVolume.
    cvol = cv.CloudVolume(opt.gs_input, mip=opt.in_mip, cache=opt.cache,
                          fill_missing=True, parallel=opt.parallel)

    # Cutout
    offset0 = cvol.mip_voxel_offset(0)
    if opt.center is not None:
        assert opt.size is not None
        opt.begin = tuple(x - (y//2) for x, y in zip(opt.center, opt.size))
        opt.end = tuple(x + y for x, y in zip(opt.begin, opt.size))
    else:
        if not opt.begin:
            opt.begin = offset0
        if not opt.end:
            if not opt.size:
                opt.end = offset0 + cvol.mip_volume_size(0)
            else:
                opt.end = tuple(x + y for x, y in zip(opt.begin, opt.size))
    sl = [slice(x,y) for x, y in zip(opt.begin, opt.end)]
    print('begin = {}'.format(opt.begin))
    print('end = {}'.format(opt.end))

    # Coordinates
    print('mip 0 = {}'.format(sl))
    sl = cvol.slices_from_global_coords(sl)
    print('mip {} = {}'.format(opt.in_mip, sl))
    cutout = cvol[sl]

    # Transpose & squeeze
    cutout = cutout.transpose([3,2,1,0])
    cutout = np.squeeze(cutout).astype(dtype)
    return cutout
