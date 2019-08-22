from __future__ import print_function

import numpy as np

import cloudvolume as cv
from cloudvolume.lib import Vec, Bbox
from taskqueue import LocalTaskQueue, MockTaskQueue
import igneous.task_creation as tc


def make_info(num_channels, layer_type, dtype, shape, resolution,
              offset=(0,0,0), chunk_size=(64,64,64)):
    return cv.CloudVolume.create_new_info(
        num_channels, layer_type, dtype, 'raw', resolution, offset, shape,
        chunk_size=chunk_size)


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


def to_tensor(data):
    """Ensure that data is a numpy 4D array."""
    assert isinstance(data, np.ndarray)
    if data.ndim == 2:
        data = data[np.newaxis,np.newaxis,...]
    elif data.ndim == 3:
        data = data[np.newaxis,...]
    elif data.ndim == 4:
        pass
    else:
        raise RuntimeError("data must be a numpy 4D array")
    assert data.ndim == 4
    return data


def ingest(data, opt):
    # Neuroglancer format
    data = to_tensor(data)
    data = data.transpose((3,2,1,0))
    num_channels = data.shape[-1]
    shape = data.shape[:-1]

    # Offset
    offset = opt.begin if opt.offset is None else opt.offset

    # Create info
    info = make_info(num_channels, opt.vol_type, str(data.dtype), shape,
                     opt.resolution, offset=offset, chunk_size=opt.chunk_size)
    print(info)
    gs_path = opt.gs_output
    print("gs_output:\n{}".format(gs_path))
    cvol = cv.CloudVolume(gs_path, mip=0, info=info, parallel=opt.parallel)
    cvol[:,:,:,:] = data
    cvol.commit_info()

    # Optional downsampling & meshing
    downsample(opt)
    mesh(opt)


def downsample(opt):
    gs_path = opt.gs_output

    # Downsample
    if opt.downsample:
        with LocalTaskQueue(parallel=opt.parallel) as tq:
            tasks = tc.create_downsampling_tasks(gs_path,
                                                 mip=0, fill_missing=True)
            tq.insert_all(tasks)


def mesh(opt):
    gs_path = opt.gs_output

    # Mesh
    if opt.mesh:
        assert opt.vol_type == 'segmentation'

        # Create mesh
        with LocalTaskQueue(parallel=opt.parallel) as tq:
            tasks = tc.create_meshing_tasks(gs_path, mip=opt.mesh_mip)
            tq.insert_all(tasks)

        # Manifest
        with MockTaskQueue() as tq:
            tasks = create_mesh_manifest_tasks(gs_path)
            tq.insert_all(tasks)
