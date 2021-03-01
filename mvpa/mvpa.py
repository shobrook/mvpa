# Standard Library
from itertools import product
from functools import partial
from multiprocessing import Pool as ProcessPool
from math import atan

# Third Party
import numpy as np
from scipy import stats
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
    spheres = []
    centers = product(*[range(radius, ub - radius) for ub in mask.shape])
    for x0, y0, z0 in centers:
        # Skip spheres where the center is outside the brain
        if not mask[x0][y0][z0]:
            continue

        spheres.append(((x0, y0, z0), _create_sphere(x0, y0, z0, radius)))

    return spheres


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


def _within_subject_mvpa(condition_A, condition_B, spheres, mask):
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

    significance_map = np.zeros_like(mask, dtype=np.float64)
    analyze_sphere = partial(
        _analyze_sphere,
        A=(A_even, A_odd),
        B=(B_even, B_odd),
        output_map=significance_map
    )

    # if n_jobs > 1 or n_jobs == -1:
    #     concurrent_exec(analyze_sphere, spheres, n_jobs)
    # else:
    #     for sphere in tqdm(spheres):
    #         analyze_sphere(*sphere)
    for sphere in tqdm(spheres):
        analyze_sphere(*sphere)

    return significance_map


def mvpa(A_set, B_set, mask, radius=2, n_jobs=1):
    """
    Parameters
    ----------
    A_set : list
        list of subjects' fMRI data for the first condition (A); format for
        each subject is a tuple, (even_trials, odd_trials), where each
        element is a numpy array with shape [num_timesteps, x, y, z]
    B_set : list
        list of subjects' fMRI data for the second condition (B); same
        format as A_set
    mask : numpy.ndarray
        boolean array with shape [x, y, z] giving locations of usable voxels
    radius : int
        radius of the searchlight sphere
    n_jobs : int
        number of CPUs to split the work up between (-1 means "all CPUs")

    Return
    ------
    t_map : numpy.ndarray
        array of t-statistic values indicating the significance of each voxel
        for condition A; same shape as the mask
    p_map : numpy.ndarray
        array of p-values associated with the t-statistics in t_map
    """

    spheres = _extract_spheres(mask, radius)
    within_subject_mvpa = partial(
        _within_subject_mvpa,
        spheres=spheres,
        mask=mask
    )

    if n_jobs > 1 or n_jobs == -1:
        sig_maps = concurrent_exec(
            within_subject_mvpa,
            zip(A_set, B_set),
            n_jobs
        )
    else:
        sig_maps = []
        for condition_A, condition_B in zip(A_set, B_set):
            sig_map = within_subject_mvpa(condition_A, condition_B)
            sig_maps.append(sig_map)

    sig_maps = np.concatenate([np.expand_dims(m, axis=3) for m in sig_maps], axis=-1)
    t_map = np.zeros_like(mask, dtype=np.float64)
    p_map = np.zeros_like(mask, dtype=np.float64)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                if not mask[x][y][z]:
                    continue

                # Significance values from each subject for one voxel
                sample = sig_maps[x][y][z]
                test = stats.ttest_1samp(sample, popmean=0.0)
                t_map[x][y][z] = test.statistic
                p_map[x][y][z] = test.pvalue

    return t_map, p_map


if __name__ == "__main__":
    import pickle
    import os
    from nilearn.image import get_data, concat_imgs, mean_img, new_img_like
    from nilearn.masking import compute_epi_mask
    from nilearn.plotting import view_img

    DATA_DIR = "lf_4_subj"

    if True:
        all_fmri_imgs = []
        A_set, B_set = [], []

        print("\tLoad fMRI images")
        for subject_id in os.listdir(DATA_DIR):
            subject_dir = os.path.join(DATA_DIR, subject_id)
            if not os.path.isdir(subject_dir):
                continue

            fmri_images = [
                pickle.load(open(
                    os.path.join(subject_dir, f"{subject_id}_LR_2.pkl"),
                    "rb"
                ))["nii"],
                pickle.load(open(
                    os.path.join(subject_dir, f"{subject_id}_RL_2.pkl"),
                    "rb"
                ))["nii"],
                pickle.load(open(
                    os.path.join(subject_dir, f"{subject_id}_LR_4.pkl"),
                    "rb"
                ))["nii"],
                pickle.load(open(
                    os.path.join(subject_dir, f"{subject_id}_RL_4.pkl"),
                    "rb"
                ))["nii"]
            ]
            condition_A = [np.moveaxis(get_data(i), -1, 0) for i in fmri_images[:2]]
            condition_B = [np.moveaxis(get_data(i), -1, 0) for i in fmri_images[2:]]

            A_set.append(condition_A)
            B_set.append(condition_B)
            all_fmri_imgs.extend(fmri_images)

        print("\tComputing mask")
        mask = compute_epi_mask(mean_img(concat_imgs(all_fmri_imgs)))

        print("\tRunning searchlight")
        t_map, p_map = mvpa(A_set, B_set, get_data(mask), radius=2, n_jobs=1)

        print("\tPickling results")
        pickle.dump(mask, open("mask.pkl", "wb"))
        pickle.dump(t_map, open("t_map.pkl", "wb"))
        pickle.dump(p_map, open("p_map.pkl", "wb"))
    else:
        print("\tLoading pickled results")
        mask = pickle.load(open("mask.pkl", "rb"))
        t_map = pickle.load(open("t_map.pkl", "rb"))
        p_map = pickle.load(open("p_map.pkl", "rb"))

    # TODO: Filter t-values by p-values < 0.05
    # t_map[np.argwhere(p_map > 0.05)] = 0.0

    print("\tPlotting t-map")
    view_img(
        new_img_like(mask, t_map),
        title="t-map",
        display_mode="z",
        black_bg=True
    ).open_in_browser()
