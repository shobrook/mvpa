# Standard Library
from itertools import product
from functools import partial
from multiprocessing import Pool as ProcessPool
from math import atan

# Third Party
import numpy as np
from tqdm import tqdm


###########
# UTILITIES
###########


def concurrent_exec(func, iterable, n_processes=-1):
    """
    Executes a function on a list of inputs in parallel.

    Parameters
    ----------
    func : callable
        function to execute
    iterable : array
        list of inputs to map the function to
    n_processes : int or None
        number of worker processes to use; if -1, then the number returned by
        os.cpu_count() is used

    Return
    ------
    list : function outputs corresponding to each input
    """

    pool = ProcessPool(processes=None if n_processes == -1 else n_processes)
    map_of_items = pool.starmap(func, iterable)
    pool.close()
    pool.join()

    return map_of_items


#########
# HELPERS
#########


def _create_sphere(x0, y0, z0, radius):
    indices = []
    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            for z in range(z0 - radius, z0 + radius + 1):
                dist = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
                if dist < 0: # Outside the sphere
                    continue

                indices.append((x, y, z))

    return np.ix_(*zip(*indices))


def _extract_spheres(mask, radius):
    centers = product(*[range(radius, ub - radius) for ub in mask.shape])
    for x0, y0, z0 in centers:
        # Skip spheres where the center is outside the brain
        if not mask[x0][y0][z0]:
            continue

        yield (x0, y0, z0), _create_sphere(x0, y0, z0, radius)


def _analyze_sphere(center, sphere, A, B, output_map):
    A_even, A_odd = A
    B_even, B_odd = B

    _A_even, _A_odd = A_even[sphere].flatten(), A_odd[sphere].flatten()
    _B_even, _B_odd = B_even[sphere].flatten(), B_odd[sphere].flatten()

    AA_sim = atan(np.corrcoef(np.vstack((_A_even, _A_odd)))[0, 1])
    BB_sim = atan(np.corrcoef(np.vstack((_B_even, _B_odd)))[0, 1])
    AB_sim = atan(np.corrcoef(np.vstack((_A_even, _B_odd)))[0, 1])
    BA_sim = atan(np.corrcoef(np.vstack((_B_even, _A_odd)))[0, 1])

    x0, y0, z0 = center
    output_map[x0][y0][z0] = AA_sim + BB_sim - AB_sim - BA_sim


######
# MAIN
######


def mvpa(condition_A, condition_B, mask, radius=2, n_jobs=1):
    """
    Parameters
    ----------
    condition_A : tuple
        subject's fMRI data for the first condition; format is
        (even_trials, odd_trials) tuple where each element is a numpy array
        with shape [num_timesteps, x, y, z]
    condition_B : tuple
        subject's fMRI data for the second condition; same format as condition_A
    mask : numpy.ndarray
        boolean array with shape [x, y, z] giving locations of usable voxels
    radius : int
        radius of the searchlight sphere
    n_jobs : int
        number of CPUs to split the work up between (-1 means 'all CPUs')

    Return
    ------
    significance_map : numpy.ndarray
        array of values indicating the significance of each voxel for
        condition A; same shape as the mask
    """

    A_even, A_odd = condition_A
    B_even, B_odd = condition_B

    A_even, A_odd = np.mean(A_even, axis=0), np.mean(A_odd, axis=0)
    B_even, B_odd = np.mean(B_even, axis=0), np.mean(B_odd, axis=0)

    spheres = _extract_spheres(mask, radius)
    significance_map = np.zeros_like(mask, dtype=np.float64)
    analyze_sphere = partial(
        _analyze_sphere,
        A=(A_even, A_odd),
        B=(B_even, B_odd),
        output_map=significance_map
    )

    if n_jobs > 1 or n_jobs == -1:
        concurrent_exec(analyze_sphere, spheres)
    else:
        num_spheres = np.prod([ub - (2 * radius) for ub in mask.shape])
        for sphere in tqdm(spheres, total=num_spheres):
            analyze_sphere(*sphere)

    return significance_map


if __name__ == "__main__":
    import pickle
    from os import path
    from nilearn.image import get_data, concat_imgs, mean_img, new_img_like
    from nilearn.masking import compute_epi_mask
    from nilearn.plotting import view_img

    if True:
        print("\tLoading fMRI images")
        fmri_images = [
            pickle.load(open("data/159744_LR_2.pkl", "rb"))["nii"],
            pickle.load(open("data/159744_RL_2.pkl", "rb"))["nii"],
            pickle.load(open("data/159744_LR_4.pkl", "rb"))["nii"],
            pickle.load(open("data/159744_RL_4.pkl", "rb"))["nii"]
        ]
        condition_A = [np.moveaxis(get_data(i), -1, 0) for i in fmri_images[:2]]
        condition_B = [np.moveaxis(get_data(i), -1, 0) for i in fmri_images[2:]]

        print("\tComputing mask")
        mask = compute_epi_mask(mean_img(concat_imgs(fmri_images)))

        print("\tRunning searchlight")
        sig_map = mvpa(condition_A, condition_B, get_data(mask), radius=2, n_jobs=1)

        print("\tPickling results")
        pickle.dump(mask, open("mask.pkl", "wb"))
        pickle.dump(sig_map, open("sig_map.pkl", "wb"))
    else:
        print("\tLoading pickled mask")
        mask = pickle.load(open("mask.pkl", "rb"))

        print("\tLoading pickled significance map")
        sig_map = pickle.load(open("sig_map.pkl", "rb"))

    print("\tPlotting significance map")
    view_img(
        new_img_like(mask, sig_map),
        title="Significance Map",
        display_mode="z",
        # cut_coords=[-9],
        # cmap="hot",
        black_bg=True
    ).open_in_browser()
