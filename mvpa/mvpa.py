# Standard Library
from itertools import product

# Third Party
import numpy as np
from tqdm import tqdm


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
    # mask is a boolean array with shape [x, y, z]

    centers = product(*[range(radius, ub - radius) for ub in mask.shape])
    for x0, y0, z0 in centers:
        # Skip spheres where the center is outside the brain
        if not mask[x0][y0][z0]:
            continue

        yield (x0, y0, z0), _create_sphere(x0, y0, z0, radius)


def mvpa(condition_A, condition_B, mask, radius):
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

    Returns
    -------
    significance_map : numpy.ndarray
        array of values indicating the significance of each voxel for
        condition A; same shape as the mask
    """

    A_even, A_odd = condition_A
    B_even, B_odd = condition_B

    A_even, A_odd = np.mean(A_even, axis=0), np.mean(A_odd, axis=0)
    B_even, B_odd = np.mean(B_even, axis=0), np.mean(B_odd, axis=0)

    significance_map = np.zeros_like(mask, dtype=np.float64)
    spheres = list(_extract_spheres(mask, radius))
    for (x0, y0, z0), sphere in tqdm(spheres):
        _A_even, _A_odd = A_even[sphere].flatten(), A_odd[sphere].flatten()
        _B_even, _B_odd = B_even[sphere].flatten(), B_odd[sphere].flatten()

        AA_sim = np.corrcoef(np.vstack((_A_even, _A_odd)))[0, 1]
        BB_sim = np.corrcoef(np.vstack((_B_even, _B_odd)))[0, 1]
        AB_sim = np.corrcoef(np.vstack((_A_even, _B_odd)))[0, 1]
        BA_sim = np.corrcoef(np.vstack((_B_even, _A_odd)))[0, 1]

        significance_map[x0][y0][z0] = AA_sim + BB_sim - AB_sim - BA_sim

    return significance_map


if __name__ == "__main__":
    import os
    import pickle
    from nilearn.image import get_data, concat_imgs, mean_img, new_img_like
    from nilearn.masking import compute_epi_mask
    from nilearn.plotting import view_img

    if True:
        condition_A, condition_B = [], []
        images = []
        for filename in os.listdir("data"):
            fmri_img = pickle.load(open(os.path.join("data", filename), "rb"))["nii"]
            fmri_data = np.moveaxis(get_data(fmri_img), -1, 0)

            if "_2" in filename:
                condition_A.append(fmri_data)
            else:
                condition_B.append(fmri_data)

            images.append(fmri_img)

        mask = compute_epi_mask(mean_img(concat_imgs(images)))
        sig_map = mvpa(condition_A, condition_B, get_data(mask), radius=3)

        pickle.dump(mask, open("mask.pkl", "wb"))
        pickle.dump(sig_map, open("sig_map.pkl", "wb"))
    else:
        mask = pickle.load(open("mask.pkl", "rb"))
        sig_map = pickle.load(open("sig_map.pkl", "rb"))

    view_img(
        new_img_like(mask, sig_map),
        title="Significance Map",
        display_mode="z",
        # cut_coords=[-9],
        # cmap="hot",
        black_bg=True
    )
