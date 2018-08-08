import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
import datetime
import vtk
from mpl_toolkits.mplot3d import Axes3D
import glob
import os


def load_vtk(vtk_path):
    """
    Loads a VTK-file.
    """

    reader = vtk.vtkDataSetReader()
    reader.SetFileName(vtk_path)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()

    points = np.zeros((data.GetNumberOfPoints(), 3))

    for i in range(data.GetNumberOfPoints()):
        points[i] = data.GetPoint(i)

    return points


def render_pointcloud(points, title=None):
    """
    Renders a point-cloud.
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:,0], points[:,1], points[:,2], s=0.5, cmap="gray", alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if title != None:
        plt.title(title)

    plt.show()
    plt.close()


def ensure_voxelgrid_shape(voxelgrid, voxelgrid_target_shape):
    voxelgrid = pad_voxelgrid(voxelgrid, voxelgrid_target_shape)
    voxelgrid = crop_voxelgrid(voxelgrid, voxelgrid_target_shape)
    return voxelgrid


def pad_voxelgrid(voxelgrid, voxelgrid_target_shape):

    pad_before = [0.0] * 3
    pad_after = [0.0] * 3
    for i in range(3):
        pad_before[i] = (voxelgrid_target_shape[i] - voxelgrid.shape[i]) // 2
        pad_before[i] = max(0, pad_before[i])
        pad_after[i] = voxelgrid_target_shape[i] - pad_before[i] - voxelgrid.shape[i]
        pad_after[i] = max(0, pad_after[i])
    voxelgrid = np.pad(
        voxelgrid,
        [(pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])],
        'constant', constant_values=[(0, 0), (0, 0), (0, 0)]
    )

    return voxelgrid


def crop_voxelgrid(voxelgrid, voxelgrid_target_shape):

    while voxelgrid.shape[0] > voxelgrid_target_shape[0]:
        voxels_start = np.count_nonzero(voxelgrid[0,:,:] != 0.0)
        voxels_end = np.count_nonzero(voxelgrid[-1,:,:] != 0.0)
        if voxels_start > voxels_end:
            voxelgrid = voxelgrid[:-1,:,:]
        else:
            voxelgrid = voxelgrid[1:,:,:]

    while voxelgrid.shape[1] > voxelgrid_target_shape[1]:
        voxels_start = np.count_nonzero(voxelgrid[:,0,:] != 0.0)
        voxels_end = np.count_nonzero(voxelgrid[:,-1,:] != 0.0)
        if voxels_start > voxels_end:
            voxelgrid = voxelgrid[:,:-1,:]
        else:
            voxelgrid = voxelgrid[:,1:,:]

    while voxelgrid.shape[2] > voxelgrid_target_shape[2]:
        voxels_start = np.count_nonzero(voxelgrid[:,:,0] != 0.0)
        voxels_end = np.count_nonzero(voxelgrid[:,:,-1] != 0.0)
        if voxels_start > voxels_end:
            voxelgrid = voxelgrid[:,:,:-1]
        else:
            voxelgrid = voxelgrid[:,:,1:]

    return voxelgrid


def center_crop_voxelgrid(voxelgrid, voxelgrid_target_shape):

    # Center crop.
    crop_start = [0.0] * 3
    crop_end = [0.0] * 3
    for i in range(3):
        crop_start[i] = (voxelgrid.shape[i] - voxelgrid_target_shape[i]) // 2
        crop_start[i] = max(0, crop_start[i])
        crop_end[i] = target_shape[i] + crop_start[i]
    voxelgrid = voxelgrid[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]

    return voxelgrid


def render_voxelgrid(voxelgrid, title=None):
    """
    Renders a voxel-grid.
    """

    figsize = (5, 5)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    transformed_voxelgrid = np.flip(np.flip(voxelgrid, axis=2), axis=0)

    facecolors = np.zeros(transformed_voxelgrid.shape + (3,))
    for x, y, z in itertools.product(range(transformed_voxelgrid.shape[0]), range(transformed_voxelgrid.shape[1]), range(transformed_voxelgrid.shape[2])):
        color = (1.0 - y / 32)
        facecolors[x, y, z, 0] = color
        facecolors[x, y, z, 1] = color
        facecolors[x, y, z, 2] = color

    ax.voxels(transformed_voxelgrid, facecolors=facecolors, edgecolor="k")

    if title != None:
        plt.title(title)

    plt.show()
    plt.close()


def get_datetime_string():
    """
    Returns a datetime string.
    """

    return datetime.datetime.now().strftime("%Y%m%d-%H%M")


def get_latest_preprocessed_dataset(path=".", filter=""):
    """
    Retrieves the path of the latest preprocessed dataset. Takes into account a filter.
    """
    glob_search_path = os.path.join(path, "*.p")
    paths = [x for x in glob.glob(glob_search_path) if filter in x]
    if len(paths) == 0:
        raise Exception("No datasets found for filter", filter)
    return sorted(paths)[-1]


def get_latest_model(path=".", filter=""):
    """
    Retrieves the path of the latest preprocessed dataset. Takes into account a filter.
    """
    glob_search_path = os.path.join(path, "*.h5")
    paths = [x for x in glob.glob(glob_search_path) if filter in x]
    if len(paths) == 0:
        raise Exception("No models found for filter", filter)
    return sorted(paths)[-1]
