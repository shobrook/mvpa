# Standard Library
from os import mkdir
from os.path import expanduser, isdir, join
from itertools import product
from functools import partial
from multiprocessing import Pool as ProcessPool
from math import atan

# Third Party
import numpy as np
from scipy import stats
from scipy.interpolate import NearestNDInterpolator
from nilearn.image import get_data, load_img, new_img_like, mean_img, concat_imgs
from nilearn.masking import compute_epi_mask
from tqdm import tqdm


#########
# HELPERS
#########


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
    map_of_items = pool.map(func, iterable)
    pool.close()
    pool.join()

    return map_of_items


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


def _extract_spheres(mask, radius, interpolate):
    step = 2 if interpolate else 1
    centers = np.transpose(np.nonzero(mask))
    spheres = []
    for center in centers[::step]:
        x0, y0, z0 = center
        # Skip spheres where the center is outside the brain
        if not mask[x0][y0][z0]:
            continue

        spheres.append(((x0, y0, z0), _create_sphere(x0, y0, z0, radius)))

    return spheres


def _filename(data_dir, subject_id):
    if not data_dir:
        home = expanduser("~")
        data_dir = join(home, "mvpa")
    if not isdir(data_dir):
        mkdir(data_dir)

    filename = join(data_dir, f"{subject_id}_scores.nii")
    return filename


######
# MAIN
######


def analyze_subject(subject_data, spheres, interpolate, mask, data_dir=None):
    """
    Parameters
    ----------
    subject_data : dict
        TODO
    spheres : list
        TODO
    interpolate : bool
        whether or not to skip every other sphere (for speed) and interpolate
        the results
    mask : numpy.ndarray
        boolean array with shape [x, y, z] giving locations of usable voxels
    data_dir : string
        path to directory of where to store MVPA results

    Return
    ------
    scores : numpy.ndarray
        array of values indicating the significance of each voxel for
        condition A; same shape as the mask
    """

    subject_id = subject_data["subject_id"]
    A_even, A_odd = subject_data["A_even_trials"], subject_data["A_odd_trials"]
    B_even, B_odd = subject_data["B_even_trials"], subject_data["B_odd_trials"]

    if all(isinstance(img, str) for img in [A_even, A_odd, B_even, B_odd]):
        A_even, A_odd = load_img(A_even), load_img(A_odd)
        B_even, B_odd = load_img(B_even), load_img(B_odd)

    A_even = np.mean(np.moveaxis(get_data(A_even), -1, 0), axis=0)
    A_odd = np.mean(np.moveaxis(get_data(A_odd), -1, 0), axis=0)
    B_even = np.mean(np.moveaxis(get_data(B_even), -1, 0), axis=0)
    B_odd = np.mean(np.moveaxis(get_data(B_odd), -1, 0), axis=0)

    _mask = get_data(mask)
    scores = np.zeros_like(_mask, dtype=np.float64)
    X, y = [], []
    for (x0, y0, z0), sphere in tqdm(spheres):
        _A_even, _A_odd = A_even[sphere].flatten(), A_odd[sphere].flatten()
        _B_even, _B_odd = B_even[sphere].flatten(), B_odd[sphere].flatten()

        AA_sim = atan(np.corrcoef(np.vstack((_A_even, _A_odd)))[0, 1])
        BB_sim = atan(np.corrcoef(np.vstack((_B_even, _B_odd)))[0, 1])
        AB_sim = atan(np.corrcoef(np.vstack((_A_even, _B_odd)))[0, 1])
        BA_sim = atan(np.corrcoef(np.vstack((_B_even, _A_odd)))[0, 1])

        scores[x0][y0][z0] = AA_sim + BB_sim - AB_sim - BA_sim

        X.append(np.array([x0, y0, z0]))
        y.append(scores[x0][y0][z0])

    if interpolate:
        interp = NearestNDInterpolator(np.vstack(X), y)
        for indices in np.transpose(np.nonzero(_mask)):
            x, y, z = indices
            if not scores[x][y][z]:
                scores[x][y][z] = interp(indices)

    filename = _filename(data_dir, subject_id)
    scores = new_img_like(mask, scores)
    scores.to_filename(filename)

    return filename


def mvpa(dataset, mask=None, radius=2, interpolate=False, n_jobs=1, data_dir=None):
    """
    Parameters
    ----------
    dataset : list
        list of subjects' fMRI data; each item in the list is a dictionary with
        the format
            {
                "subject_id": "...",
                "A_even_trials": "...",
                "A_odd_trials": "...",
                "B_even_trials": "...",
                "B_odd_trials": "..."
            }
        where "A/B_even/odd_trials" holds a niimg or path (string) to a NIfTI
        file representing the image data for the trials for that condition (A
        or B)
    mask : Niimg-like object
        boolean image giving location of voxels containing usable signals
    radius : int
        radius of the searchlight sphere
    interpolate : bool
        whether or not to skip every other sphere (for speed) and interpolate
        the results
    n_jobs : int
        number of CPUs to split the work up between (-1 means "all CPUs")
    data_dir : string
        path to directory to store MVPA results

    Return
    ------
    score_map_fpaths : list
        list of paths to NIfTI files representing the MVPA scores for each
        subject
    """

    if not mask:
        niimgs = []
        for subject_data in dataset:
            niimgs.extend([
                subject_data["A_even_trials"],
                subject_data["A_odd_trials"],
                subject_data["B_even_trials"],
                subject_data["B_odd_trials"]
            ])
        mask = compute_epi_mask(mean_img(concat_imgs(niimgs)))

    spheres = _extract_spheres(get_data(mask), radius, interpolate)
    _analyze_subject = partial(
        analyze_subject,
        spheres=spheres,
        interpolate=interpolate,
        mask=mask,
        data_dir=data_dir
    )

    if n_jobs > 1 or n_jobs == -1:
        score_map_fpaths = concurrent_exec(
            _analyze_subject,
            dataset,
            n_jobs
        )
    else:
        score_map_fpaths = []
        for subject_data in dataset:
            score_map_fpath = _analyze_subject(subject_data)
            score_map_fpaths.append(score_map_fpath)

    return score_map_fpaths


def significance_map(subject_scores, mask):
    """
    Parameters
    ----------
    subject_scores : list
        list of niimgs or paths to NIfTI files representing the MVPA scores for
        each subject
    mask : Niimg-like object
        boolean image giving location of voxels containing usable signals

    Return
    ------
    t_map : numpy.ndarray
        array of t-statistic values indicating the significance of each voxel
        for condition A; same shape as the mask
    p_map : numpy.ndarray
        array of p-values associated with the t-statistics in t_map
    """

    score_maps = []
    for score_map in subject_scores:
        if isinstance(score_map, str):
            score_map = get_data(load_img(score_map))
        else:
            score_map = get_data(score_map)

        score_maps.append(np.expand_dims(score_map, axis=3))

    mask = np.mean(score_maps, axis=0).astype(bool)
    score_maps = np.concatenate(score_maps, axis=-1)
    t_map = np.zeros_like(mask, dtype=np.float64)
    p_map = np.zeros_like(mask, dtype=np.float64)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                if not mask[x][y][z]:
                    continue

                # Significance values from each subject for one voxel
                sample = score_maps[x][y][z]
                test = stats.ttest_1samp(sample, popmean=0.0)
                t_map[x][y][z] = test.statistic
                p_map[x][y][z] = test.pvalue

    return t_map, p_map


if __name__ == "__main__":
    import pickle
    from os import listdir
    from os.path import isfile
    from nilearn.plotting import view_img, plot_stat_map

    DATA_DIR = "lf_4_subj"

    if False:
        print("\tLoading subject NIIMG file paths")
        dataset, niimgs = [], []
        for subject_id in listdir(DATA_DIR):
            subject_dir = join(DATA_DIR, subject_id)
            if not isdir(subject_dir):
                continue

            for filename in listdir(subject_dir):
                if not filename.endswith(".pkl"):
                    continue

                niimg = pickle.load(open(join(subject_dir, filename), "rb"))["nii"]
                niimg.to_filename(join(subject_dir, f"{filename[:-4]}.nii"))

            subject_data = {
                "subject_id": subject_id,
                "A_even_trials": join(subject_dir, f"{subject_id}_LR_2.nii"),
                "A_odd_trials": join(subject_dir, f"{subject_id}_RL_2.nii"),
                "B_even_trials": join(subject_dir, f"{subject_id}_LR_4.nii"),
                "B_odd_trials": join(subject_dir, f"{subject_id}_RL_4.nii")
            }
            dataset.append(subject_data)
            niimgs.extend([
                subject_data["A_even_trials"],
                subject_data["A_odd_trials"],
                subject_data["B_even_trials"],
                subject_data["B_odd_trials"]
            ])

        print("\tCreating mask")
        mask = compute_epi_mask(mean_img(concat_imgs(niimgs)))
        mask.to_filename("mask.nii")

        print("\tRunning searchlight")
        score_maps = mvpa(dataset, mask, radius=2, interpolate=True, n_jobs=1, data_dir="score_maps")
    else:
        print("\tLoading subject scores")
        score_maps = [join("score_maps", f) for f in listdir("score_maps") if isfile(join("score_maps", f))]

        print("\tLoading mask")
        mask = load_img("mask.nii")

    print("\tCreating significance map (t-map, p-map)")
    t_map, p_map = significance_map(score_maps, mask)

    # TODO: Filter t-values by p-values < 0.05
    # t_map[np.argwhere(p_map > 0.05)] = 0.0
    p_map[p_map > 0.05] = 0.0

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plot_stat_map(
        new_img_like(mask, p_map),
        colorbar=True,
        display_mode="z",
        figure=fig
    )
    plt.show()

    print("\tPlotting t-map")
    view_img(
        new_img_like(mask, t_map),
        title="t-map",
        display_mode="z",
        black_bg=True
    ).open_in_browser()
