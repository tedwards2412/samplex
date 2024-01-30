import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage
import itertools

plt.style.use(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.mplstyle")
)

# purple - green - darkgoldenrod - blue - red
colors = ["purple", "#306B37", "darkgoldenrod", "#3F7BB6", "#BF4145"]

################################################################################


def ctr_level2d(histogram2d, lvl, infinite=False):
    hist = histogram2d.flatten() * 1.0
    hist.sort()
    cum_hist = np.cumsum(hist[::-1])
    cum_hist /= cum_hist[-1]

    alvl = np.searchsorted(cum_hist, lvl)[::-1]
    clist = [0] + [hist[-i] for i in alvl] + [hist.max()]
    if not infinite:
        return clist[1:]
    return clist


def get_hist(data, num_bins=30, weights=[None]):
    if not any(weights):
        weights = np.ones(len(data))
    hist, bin_edges = np.histogram(data, bins=num_bins, weights=weights)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return hist, bin_edges, bin_centres


def get_hist2d(datax, datay, num_bins=[30, 30], weights=[None]):
    if not any(weights):
        weights = np.ones(len(datax))
    hist, bin_edgesx, bin_edgesy = np.histogram2d(
        datax, datay, bins=num_bins, weights=weights
    )
    bin_centresx = 0.5 * (bin_edgesx[1:] + bin_edgesx[:-1])
    bin_centresy = 0.5 * (bin_edgesy[1:] + bin_edgesy[:-1])
    hist2 = hist.min() + np.zeros((hist.shape[0] + 4, hist.shape[1] + 4))
    hist2[2:-2, 2:-2] = hist
    hist2[2:-2, 1] = hist[:, 0]
    hist2[2:-2, -2] = hist[:, -1]
    hist2[1, 2:-2] = hist[0]
    hist2[-2, 2:-2] = hist[-1]
    hist2[1, 1] = hist[0, 0]
    hist2[1, -2] = hist[0, -1]
    hist2[-2, 1] = hist[-1, 0]
    hist2[-2, -2] = hist[-1, -1]
    bin_centresx2 = np.concatenate(
        [
            bin_centresx[0] + np.array([-2, -1]) * np.diff(bin_centresx[:2]),
            bin_centresx,
            bin_centresx[-1] + np.array([1, 2]) * np.diff(bin_centresx[-2:]),
        ]
    )
    bin_centresy2 = np.concatenate(
        [
            bin_centresy[0] + np.array([-2, -1]) * np.diff(bin_centresy[:2]),
            bin_centresy,
            bin_centresy[-1] + np.array([1, 2]) * np.diff(bin_centresy[-2:]),
        ]
    )
    return hist2, bin_edgesx, bin_edgesy, bin_centresx2, bin_centresy2


def plot_hist(data, ax, num_bins=30, weights=[None], color=None, zorder=None):
    if not any(weights):
        weights = np.ones(len(data))
    if color == None:
        color = "darkblue"

    hist, bin_edges, bin_centres = get_hist(data, num_bins=num_bins, weights=weights)
    ax.step(bin_centres, hist / max(hist), where="mid", color=color, zorder=zorder)
    ax.set_ylim(ymax=1.05)


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_hist2d(
    datax,
    datay,
    ax,
    num_bins=[30, 30],
    weights=[None],
    color=None,
    zorder=0,
    interpolation_smoothing=3.0,
    gaussian_smoothing=0.5,
):
    if not any(weights):
        weights = np.ones(len(datax))
    if color == None:
        color = "black"

    hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy = get_hist2d(
        datax, datay, num_bins=num_bins, weights=weights
    )

    sigma = interpolation_smoothing * gaussian_smoothing

    interp_y_centers = scipy.ndimage.zoom(
        bin_centresy, interpolation_smoothing, mode="reflect"
    )
    interp_x_centers = scipy.ndimage.zoom(
        bin_centresx, interpolation_smoothing, mode="reflect"
    )
    interp_hist = scipy.ndimage.zoom(hist, interpolation_smoothing, mode="reflect")
    interp_smoothed_hist = scipy.ndimage.filters.gaussian_filter(
        interp_hist, [sigma, sigma], mode="reflect"
    )

    ax.contourf(
        interp_x_centers,
        interp_y_centers,
        np.transpose(interp_smoothed_hist),
        colors=[adjust_lightness(color, 1.4), adjust_lightness(color, 0.8)],
        levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]),
        zorder=zorder,
        alpha=0.55,
    )
    ax.contour(
        interp_x_centers,
        interp_y_centers,
        np.transpose(interp_smoothed_hist),
        colors=[color, adjust_lightness(color, 0.8)],
        linewidths=1.5,
        levels=ctr_level2d(interp_smoothed_hist.copy(), [0.68, 0.95]),
        zorder=zorder,
    )


def corner_plot(
    data,
    names,
    num_bins=[None],
    weights=[None],
    lims=[None],
    color=None,
    labelsize=None,
    ticksize=None,
    zorder=None,
    fig=None,
    axes=None,
    interpolation_smoothing=3.0,
    gaussian_smoothing=0.5,
):
    size = len(data[0, :])
    positions = list(itertools.combinations(range(size), 2))

    if not any(weights):
        weights = np.ones(len(data[:, 0]))
    if not any(lims):
        lims = [
            [data[:, pos].min(), data[:, pos].max()] for pos in range(len(data[0, :]))
        ]
    if not any(num_bins):
        num_bins = [30] * size
    if color is None:
        color = colors[4]
    if labelsize is None:
        labelsize = 18
    if ticksize is None:
        ticksize = 16
    if zorder is None:
        zorder = 0

    new_fig = False
    if fig is None:
        fig = plt.figure()
        axes = dict()
        new_fig = True

    # 2D posteriors
    for posy, posx in positions:
        datax = data[:, posx]
        datay = data[:, posy]

        if new_fig:
            ax = plt.subplot2grid((size, size), (posx, posy))
            axes[str(posx) + str(posy)] = ax

            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_xlim(xmin=lims[posy][0], xmax=lims[posy][1])
            ax.set_ylim(ymin=lims[posx][0], ymax=lims[posx][1])

            if posx == size - 1:
                ax.set_xlabel(names[posy], fontsize=labelsize)
                ax.xaxis.set_label_coords(0.5, -0.35)
                ax.tick_params(axis="x", which="both", labelsize=ticksize, rotation=45)
                if posy != 0:
                    ax.axes.yaxis.set_ticklabels([])
            if posy == 0:
                ax.set_ylabel(names[posx], fontsize=labelsize)
                ax.yaxis.set_label_coords(-0.35, 0.5)
                ax.tick_params(axis="y", which="both", labelsize=ticksize, rotation=45)
            if posx != size - 1:
                if posy != 0:
                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                else:
                    ax.axes.xaxis.set_ticklabels([])
        else:
            ax = axes[str(posx) + str(posy)]

        plot_hist2d(
            datay,
            datax,
            ax,
            [num_bins[posy], num_bins[posx]],
            weights,
            color,
            zorder,
            interpolation_smoothing,
            gaussian_smoothing,
        )

    # 1D posteriors
    for pos in range(size):
        if new_fig:
            ax = plt.subplot2grid((size, size), (pos, pos))
            axes[str(pos) + str(pos)] = ax
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticks([])
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_xlim(xmin=lims[pos][0], xmax=lims[pos][1])

            if pos != size - 1:
                ax.tick_params(axis="x", which="both", labelbottom=False)
            else:
                ax.set_xlabel(names[pos], fontsize=labelsize)
                ax.xaxis.set_label_coords(0.5, -0.35)
                ax.tick_params(axis="x", which="both", labelsize=ticksize, rotation=45)
        else:
            ax = axes[str(pos) + str(pos)]

        plot_hist(data[:, pos], ax, num_bins[pos], weights, color, zorder)

    return fig, axes
