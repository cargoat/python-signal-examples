import numpy as np

# -----------------------------------------------------------------------------------------
def count_pixels (img):
    # get the set of unique pixel values and their corresponding indices and
    # counts
    values, bin_idx, counts = np.unique(img, return_inverse=True, return_counts=True)
    return values, bin_idx, counts

# -----------------------------------------------------------------------------------------
def pdf (img):
    values, bin_idx, counts = count_pixels(img)
    counts = counts.astype(np.float64) / np.sum(img.shape)
    return values, counts

# -----------------------------------------------------------------------------------------
def cdf (img):
    values, bin_idx, counts = count_pixels (img)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    quantiles = np.cumsum(counts).astype(np.float64)
    quantiles /= quantiles[-1]

    return values, quantiles
# -----------------------------------------------------------------------------------------
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    from PIL import Image  
    import os
    import matplotlib.pyplot as plt

    here = os.path.dirname(os.path.realpath(__file__))

    source = np.array (Image.open (os.path.join (here, 'lena_gray_512.tif')).convert('L'))
    reference = np.array (Image.open (os.path.join (here, 'flowers.tif')).convert('L'))
    
    matched = hist_match (source, reference)
  
    s_val, s_count = pdf(source)
    r_val, r_count = pdf(reference)
    m_val, m_count = pdf(matched)
    
    _, s_cdf = cdf (source)
    _, r_cdf = cdf (reference)
    _, m_cdf = cdf (matched)

    plt.figure()
    plt.plot (s_val, s_cdf, 'b.-', label='source')
    plt.plot (r_val, r_cdf, 'r.-', label='reference')
    plt.plot (m_val, m_cdf, 'y.-', label='matched')
    plt.title ("CDF of Images")
    ax = plt.gca()
    ax.legend()

    plt.figure()
    plt.plot (s_val, s_count, 'b.-', label='source')
    plt.plot (r_val, r_count, 'r.-', label='reference')
    plt.plot (m_val, m_count, 'y.-', label='matched')
    plt.title ("PDF (histogram) of Images")
    ax = plt.gca()
    ax.legend()

    plt.figure ()
    plt.imshow (source, 'gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.title ("Original")

    plt.figure ()
    plt.imshow (reference, 'gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.title ("Reference")

    plt.figure ()
    plt.imshow (matched, 'gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.title ("Matched")

    plt.show()