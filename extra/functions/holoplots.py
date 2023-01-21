import holoviews as hv
import datashader
import xarray as xr
import numpy as np
from holoviews.operation.datashader import regrid


def normalize(agg,max):
    min_val = 0
    max_val = max
    c = 25
    th = .11
    range_val = max_val - min_val
    norm = (agg - min_val) / range_val 
    norm = 1 / (1 + np.exp(c * (th - norm))) 
    norm = norm * 255
    return norm


def combine_bands(r: xr.DataArray, g: xr.DataArray, b: xr.DataArray, gamma: float = .8):
    rmax = r.max().values.item() * gamma
    gmax = g.max().values.item() * gamma
    bmax = b.max().values.item() * gamma
    xs, ys = r['x'], r['y']
    r, g, b = [datashader.utils.orient_array(img) for img in (r, g, b)]

    a = (np.where(np.isnan(b),0,255)).astype(np.uint8)
    r = (normalize(r,rmax)).astype(np.uint8)  # try it to return valid RGB (0-255 range)
    g = (normalize(g,gmax)).astype(np.uint8)
    b = (normalize(b,bmax)).astype(np.uint8)
    return hv.RGB((xs, ys[::-1], r, g, b, a), vdims=list('RGBA'))


def plot_single_hv(ds: xr.Dataset, variables: list = ['red', 'green', 'blue'], **kwargs):
    """
    2d image plot using holoviews. 2d array only, not 3d (discard time or other coord before using)
    only the first three variables will be used
    """
    if len(variables) != 3:
        raise Exception('The number of data variables provided, must be 3')
    
    r=ds[variables[0]]
    g=ds[variables[1]]
    b=ds[variables[2]]

    true_color = combine_bands(r,g,b, **kwargs)
    return regrid(true_color)