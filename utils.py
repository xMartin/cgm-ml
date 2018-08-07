import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
import datetime
import vtk
from mpl_toolkits.mplot3d import Axes3D
import glob


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


def get_latest_preprocessed_dataset(filter):
    """
    Retrieves the path of the latest preprocessed dataset. Takes into account a filter.
    """
    paths = [x for x in glob.glob("*.p") if filter in x]
    if len(paths) == 0:
        raise Exception("No datasets found for filter", filter)
    return sorted(paths)[-1]


def get_latest_model(filter):
    """
    Retrieves the path of the latest preprocessed dataset. Takes into account a filter.
    """
    paths = [x for x in glob.glob("*.h5") if filter in x]
    if len(paths) == 0:
        raise Exception("No models found for filter", filter)
    return sorted(paths)[-1]
