# Standard Library
from os import mkdir
from os.path import expanduser, isdir, join
from multiprocessing import Pool as ProcessPool

# Third Party
import numpy as np
from nilearn.image import index_img, mean_img


#####################
# SEARCHLIGHT SPHERES
#####################


def _create_sphere(x0, y0, z0, radius):
    """
    Creates a sphere of a given radius with a given center.

    Parameters
    ----------
    x0 : int
        x-coordinate (index) of the sphere's center
    y0 : int
        y-coordinate (index) of the sphere's center
    z0 : int
        z-coordinate (index) of the sphere's center
    radius : int
        radius (in voxels) of the sphere

    Returns
    -------
    sphere_index_arrays : tuple of numpy.ndarrays
        indices that define the sphere
    """

    indices = []
    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            for z in range(z0 - radius, z0 + radius + 1):
                dist = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
                if dist < 0: # Outside the sphere
                    continue

                indices.append((x, y, z))

    sphere_index_arrays = np.ix_(*zip(*indices))
    return sphere_index_arrays


def extract_spheres(mask, radius, interpolate):
    """
    Extracts spheres of a given radius from a mask.

    Parameters
    ----------
    mask : numpy.ndarray
        boolean image giving location of voxels containing usable signals
    radius : int
        radius of the sphere
    interpolate : bool
        whether or not to skip every other sphere

    Returns
    -------
    spheres : list
        list of (center, sphere) tuples, where center is a tuple of coordinates
        of the sphere's center, and sphere is a set of indices defining the
        sphere
    """

    is_out_of_bounds = lambda i, ub: i + radius >= ub or i - radius < 0
    step = 2 if interpolate else 1
    centers = np.transpose(np.nonzero(mask))

    spheres = []
    for center in centers[::step]:
        x0, y0, z0 = center
        # Skip spheres where the center is outside the brain
        if not mask[x0][y0][z0]:
            continue

        if any(is_out_of_bounds(i, ub) for i, ub in zip(center, mask.shape)):
            continue

        spheres.append(((x0, y0, z0), _create_sphere(x0, y0, z0, radius)))

    return spheres


###############
# MISCELLANEOUS
###############


def concurrent_exec(func, iterable, n_processes=-1):
    """
    Executes a function on a list of inputs in parallel.

    Parameters
    ----------
    func : callable
        function to execute
    iterable : array-like
        list of inputs to map the function to
    n_processes : int, optional
        number of worker processes to use; if -1, then the number returned by
        os.cpu_count() is used

    Returns
    -------
    map_of_items : list
        function outputs corresponding to each input
    """

    pool = ProcessPool(processes=None if n_processes == -1 else n_processes)
    map_of_items = pool.starmap(func, iterable)
    pool.close()
    pool.join()

    return map_of_items


def score_map_filename(data_dir, subject_id):
    """
    Creates file path for a subject's score map.

    Parameters
    ----------
    data_dir : str
        path to data directory (where to save the score maps)
    subject_id : str
        unique ID of subject being analyzed

    Returns
    -------
    filename : str
        path to subject's score map
    """

    if not data_dir:
        home = expanduser("~")
        data_dir = join(home, "mvpa")
    if not isdir(data_dir):
        mkdir(data_dir)

    filename = join(data_dir, f"{subject_id}_scores.nii")
    return filename


def even_odd_split(runs):
    """
    Splits fMRI scan into even and odd runs, and returns the mean image for
    both sets.

    Parameters
    ----------
    runs : Niimg-like object
        4D niimg or path (string) to a NIfTI file that represents a subject's
        fMRI scan under a particular condition

    Returns
    -------
    even_runs : Niimg-like object
        3D niimg representing the even runs
    odd_runs : object
        3D niimg representing the odd runs
    """

    even_runs = mean_img(index_img(runs, range(0, runs.shape[-1], 2)))
    odd_runs = mean_img(index_img(runs, range(1, runs.shape[-1], 2)))

    return even_runs, odd_runs
