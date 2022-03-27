# Standard Library
from functools import partial
from math import atanh

# Third Party
import numpy as np
from scipy import stats
from scipy.interpolate import NearestNDInterpolator
from nilearn.image import get_data, load_img, new_img_like, mean_img, concat_imgs
from nilearn.masking import compute_epi_mask
from tqdm import tqdm

# Local Modules
from corrmvpa.utilities import extract_spheres, concurrent_exec, score_map_filename


######
# MAIN
######


def analyze_subject(subject_id, A, B, spheres, interpolate, mask, data_dir=None):
    """
    Parameters
    ----------
    subject_id : int
        unique ID of the subject (index of the fMRI data in the input dataset)
    A : tuple
        tuple of (even_trials, odd_trials) for the first condition (A);
        even/odd_trials is the subject's mean fMRI data for that trial, and
        should be either a 3D niimg or path (string) to a NIfTI file
    B : tuple
        tuple of (even_trials, odd_trials) for the second condition (B);
        formatted the same as A
    spheres : list
        TODO
    interpolate : bool
        whether or not to skip every other sphere and interpolate the results;
        used to speed up the analysis
    mask : Niimg-like object
        boolean image giving location of voxels containing usable signals
    data_dir : string
        path to directory where MVPA results should be stored

    Returns
    -------
    score_map_fpath : str
        path to NIfTI file with values indicating the significance of each voxel
        for condition A; same shape as the mask
    """

    A_even, A_odd = A
    B_even, B_odd = B

    if all(isinstance(img, str) for img in [A_even, A_odd, B_even, B_odd]):
        A_even, A_odd = load_img(A_even), load_img(A_odd)
        B_even, B_odd = load_img(B_even), load_img(B_odd)

    A_even, A_odd = get_data(A_even), get_data(A_odd)
    B_even, B_odd = get_data(B_even), get_data(B_odd)

    _mask = get_data(mask)
    scores = np.zeros_like(_mask, dtype=np.float64)
    X, y = [], []
    for (x0, y0, z0), sphere in tqdm(spheres):
        _A_even, _A_odd = A_even[sphere].flatten(), A_odd[sphere].flatten()
        _B_even, _B_odd = B_even[sphere].flatten(), B_odd[sphere].flatten()

        AA_sim = atanh(np.corrcoef(np.vstack((_A_even, _A_odd)))[0, 1])
        BB_sim = atanh(np.corrcoef(np.vstack((_B_even, _B_odd)))[0, 1])
        AB_sim = atanh(np.corrcoef(np.vstack((_A_even, _B_odd)))[0, 1])
        BA_sim = atanh(np.corrcoef(np.vstack((_B_even, _A_odd)))[0, 1])

        scores[x0][y0][z0] = AA_sim + BB_sim - AB_sim - BA_sim

        X.append(np.array([x0, y0, z0]))
        y.append(scores[x0][y0][z0])

    if interpolate:
        interp = NearestNDInterpolator(np.vstack(X), y)
        for indices in np.transpose(np.nonzero(_mask)):
            x, y, z = indices
            if not scores[x][y][z]:
                scores[x][y][z] = interp(indices)

    score_map_fpath = score_map_filename(data_dir, subject_id)
    scores = new_img_like(mask, scores)
    scores.to_filename(score_map_fpath)

    return score_map_fpath


def correlation_searchlight(A, B, mask=None, radius=2, interpolate=False, n_jobs=1, data_dir=None):
    """
    Parameters
    ----------
    A : list
        list of (even_trials, odd_trials) tuples for the first condition (A);
        each tuple represents a subject's mean fMRI data for both trials, and
        even/odd_trials should be a 3D niimg or path (string) to a NIfTI file
    B : list
        list of (even_trials, odd_trials) tuples for the second condition (B);
        formatted the same as A
    mask : Niimg-like object
        boolean image giving location of voxels containing usable signals
    radius : int
        radius of the searchlight sphere
    interpolate : bool
        whether or not to skip every other sphere and interpolate the results;
        used to speed up the analysis
    n_jobs : int
        number of CPUs to split the work up between (-1 means "all CPUs")
    data_dir : string
        path to directory where MVPA results should be stored

    Returns
    -------
    score_map_fpaths : list
        list of paths to NIfTI files representing the MVPA scores for each
        subject
    """

    if not mask:
        niimgs = [niimg for trials in A + B for niimg in trials]
        mask = compute_epi_mask(mean_img(concat_imgs(niimgs)))

    spheres = extract_spheres(get_data(mask), radius, interpolate)
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
            [(i, _A, _B) for i, (_A, _B) in enumerate(zip(A, B))],
            n_jobs
        )
    else:
        score_map_fpaths = []
        for subject_id, (_A, _B) in enumerate(zip(A, B)):
            score_map_fpath = _analyze_subject(subject_id, _A, _B)
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

    Returns
    -------
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
