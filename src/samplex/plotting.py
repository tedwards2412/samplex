import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import matplotlib

# import glob
import scipy.ndimage
import itertools

# purple - green - darkgoldenrod - blue - red
colors = ["purple", "#306B37", "darkgoldenrod", "#3F7BB6", "#BF4145"]

##########################################################################


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


def get_hist(data, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(data))
    hist, bin_edges = np.histogram(data, bins=num_bins, weights=weights)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return hist, bin_edges, bin_centres


def get_hist2d(datax, datay, num_bins=40, weights=[None]):
    if not any(weights):
        weights = np.ones(len(datax))
    hist, bin_edgesx, bin_edgesy = np.histogram2d(
        datax, datay, bins=num_bins, weights=weights
    )
    bin_centresx = 0.5 * (bin_edgesx[1:] + bin_edgesx[:-1])
    bin_centresy = 0.5 * (bin_edgesy[1:] + bin_edgesy[:-1])

    return hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy


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


def plot_hist2d(datax, datay, ax, num_bins=30, weights=[None], color=None, zorder=0):
    if not any(weights):
        weights = np.ones(len(datax))
    if color == None:
        color = "black"

    hist, bin_edgesx, bin_edgesy, bin_centresx, bin_centresy = get_hist2d(
        datax, datay, num_bins=num_bins, weights=weights
    )

    interpolation_smoothing = 3.0
    gaussian_smoothing = 0.5
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
    num_bins=30,
    weights=[None],
    lims=[None],
    color=None,
    labelsize=None,
    ticksize=None,
    zorder=None,
    fig=None,
    axes=None,
):
    if not any(weights):
        weights = np.ones(len(data[:, 0]))
    if not any(lims):
        lims = [
            [data[:, pos].min(), data[:, pos].max()] for pos in range(len(data[0, :]))
        ]
    if color is None:
        color = colors[4]
    if labelsize is None:
        labelsize = 18
    if ticksize is None:
        ticksize = 16
    if zorder is None:
        zorder = 0

    size = len(data[0, :])
    positions = list(itertools.combinations(range(size), 2))

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

        plot_hist2d(datay, datax, ax, num_bins, weights, color, zorder)

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

        plot_hist(data[:, pos], ax, num_bins, weights, color, zorder)

    return fig, axes


##########################################################################

# chains_HST = []
# chains_IllustrisTNG = []

# for filepath in glob.iglob("../../Data/UVLF_HST_ST_model1/*__*.txt"):
#     data = np.loadtxt(filepath)
#     chains_HST.append(data)
# for filepath in glob.iglob("../../Data/UVLF_IllustrisTNG_ST_model1/*__*.txt"):
#     data = np.loadtxt(filepath)
#     chains_IllustrisTNG.append(data)

# chains_HST = np.vstack(np.array(chains_HST))
# chains_IllustrisTNG = np.vstack(np.array(chains_IllustrisTNG))

# names = [
#     r"$\sigma_8$",
#     r"$\alpha_*$",
#     r"$\beta_*$",
#     r"$\epsilon_*^\mathrm{s}$",
#     r"$\epsilon_*^\mathrm{i}$",
#     r"$M_c^\mathrm{s}$",
#     r"$M_c^\mathrm{i}$",
# ]
# lims = [
#     [0.4, 1.2],
#     [-1.1, -0.2],
#     [0.0, 3.0],
#     [-3.0, 1.0],
#     [-2.8, -1.2],
#     [-3.0, 3.0],
#     [10.5, 12.7],
# ]

# fig, axes = corner_plot(
#     data=chains_HST[:, [2, 6, 7, 8, 9, 10, 11]],
#     names=names,
#     num_bins=20,
#     weights=chains_HST[:, 0],
#     lims=lims,
#     color=colors[3],
#     labelsize=22,
#     ticksize=None,
#     zorder=1,
#     fig=None,
#     axes=None,
# )
# corner_plot(
#     data=chains_IllustrisTNG[:, [2, 6, 7, 8, 9, 10, 11]],
#     names=names,
#     num_bins=20,
#     weights=chains_IllustrisTNG[:, 0],
#     lims=lims,
#     color=colors[-1],
#     labelsize=22,
#     ticksize=None,
#     zorder=0,
#     fig=fig,
#     axes=axes,
# )

# patch_blue = mpatches.Patch(color=colors[3], lw=1.5, label=r"$\mathrm{HST}$", alpha=0.8)
# patch_yellow = mpatches.Patch(
#     color=colors[-1], lw=1.5, label=r"$\mathrm{IllustrisTNG}$", alpha=0.8
# )
# plt.legend(
#     handles=[patch_blue, patch_yellow],
#     loc="upper right",
#     bbox_to_anchor=(-0.3, 6.2),
#     prop={"size": 23},
# )

# axes[str(6) + str(0)].set_xticks([0.5, 0.7, 0.9, 1.1])
# axes[str(6) + str(0)].set_xticklabels([r"$0.5$", r"$0.7$", r"$0.9$", r"$1.1$"])
# axes[str(6) + str(0)].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
# for num in range(7):
#     axes[str(num) + str(0)].set_xticks([0.5, 0.7, 0.9, 1.1])
#     axes[str(num) + str(0)].xaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(4)
#     )

# axes[str(1) + str(0)].set_yticks([-1.0, -0.7, -0.4])
# axes[str(1) + str(0)].set_yticklabels([r"$-1.0$", r"$-0.7$", r"$-0.4$"])
# axes[str(1) + str(0)].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
# axes[str(6) + str(1)].set_xticks([-1.0, -0.7, -0.4])
# axes[str(6) + str(1)].set_xticklabels([r"$-1.0$", r"$-0.7$", r"$-0.4$"])
# axes[str(6) + str(1)].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))
# for num in range(5):
#     num += 1
#     axes[str(num) + str(1)].set_xticks([-1.0, -0.7, -0.4])
#     axes[str(num) + str(1)].xaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(3)
#     )

# axes[str(2) + str(0)].set_yticks([0.4, 1.2, 2.0, 2.8])
# axes[str(2) + str(0)].set_yticklabels([r"$0.4$", r"$1.2$", r"$2.0$", r"$2.8$"])
# axes[str(2) + str(0)].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
# axes[str(6) + str(2)].set_xticks([0.4, 1.2, 2.0, 2.8])
# axes[str(6) + str(2)].set_xticklabels([r"$0.4$", r"$1.2$", r"$2.0$", r"$2.8$"])
# axes[str(6) + str(2)].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
# for num in range(4):
#     num += 2
#     axes[str(num) + str(2)].set_xticks([0.4, 1.2, 2.0, 2.8])
#     axes[str(num) + str(2)].xaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(4)
#     )
# for num in range(2):
#     axes[str(2) + str(num)].set_yticks([0.4, 1.2, 2.0, 2.8])
#     axes[str(2) + str(num)].yaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(4)
#     )

# axes[str(3) + str(0)].set_yticks([-2.5, -1.5, -0.5, 0.5])
# axes[str(3) + str(0)].set_yticklabels([r"$-2.5$", r"$-1.5$", r"$-0.5$", r"$0.5$"])
# axes[str(3) + str(0)].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
# axes[str(6) + str(3)].set_xticks([-2.5, -1.5, -0.5, 0.5])
# axes[str(6) + str(3)].set_xticklabels([r"$-2.5$", r"$-1.5$", r"$-0.5$", r"$0.5$"])
# axes[str(6) + str(3)].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
# for num in range(3):
#     num += 3
#     axes[str(num) + str(3)].set_xticks([-2.5, -1.5, -0.5, 0.5])
#     axes[str(num) + str(3)].xaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(5)
#     )
# for num in range(3):
#     axes[str(3) + str(num)].set_yticks([-2.5, -1.5, -0.5, 0.5])
#     axes[str(3) + str(num)].yaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(5)
#     )

# axes[str(4) + str(0)].set_yticks([-2.5, -2.0, -1.5])
# axes[str(4) + str(0)].set_yticklabels([r"$-2.5$", r"$-2.0$", r"$-1.5$"])
# axes[str(4) + str(0)].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
# axes[str(6) + str(4)].set_xticks([-2.5, -2.0, -1.5])
# axes[str(6) + str(4)].set_xticklabels([r"$-2.5$", r"$-2.0$", r"$-1.5$"])
# axes[str(6) + str(4)].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
# for num in range(2):
#     num += 4
#     axes[str(num) + str(4)].set_xticks([-2.5, -2.0, -1.5])
#     axes[str(num) + str(4)].xaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(5)
#     )
# for num in range(4):
#     axes[str(4) + str(num)].set_yticks([-2.5, -2.0, -1.5])
#     axes[str(4) + str(num)].yaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(5)
#     )

# axes[str(5) + str(0)].set_yticks([-2.0, 0.0, 2.0])
# axes[str(5) + str(0)].set_yticklabels([r"$-2$", r"$0$", r"$2$"])
# axes[str(5) + str(0)].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
# axes[str(6) + str(5)].set_xticks([-2.0, 0.0, 2.0])
# axes[str(6) + str(5)].set_xticklabels([r"$-2$", r"$0$", r"$2$"])
# axes[str(6) + str(5)].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(4))
# for num in range(1):
#     num += 5
#     axes[str(num) + str(5)].set_xticks([-2.0, 0.0, 2.0])
#     axes[str(num) + str(5)].xaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(4)
#     )
# for num in range(5):
#     axes[str(5) + str(num)].set_yticks([-2.0, 0.0, 2.0])
#     axes[str(5) + str(num)].yaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(4)
#     )

# axes[str(6) + str(0)].set_yticks([10.5, 11.5, 12.5])
# axes[str(6) + str(0)].set_yticklabels([r"$10.5$", r"$11.5$", r"$12.5$"])
# axes[str(6) + str(0)].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
# axes[str(6) + str(6)].set_xticks([10.5, 11.5, 12.5])
# axes[str(6) + str(6)].set_xticklabels([r"$10.5$", r"$11.5$", r"$12.5$"])
# axes[str(6) + str(6)].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
# for num in range(6):
#     axes[str(6) + str(num)].set_yticks([10.5, 11.5, 12.5])
#     axes[str(6) + str(num)].yaxis.set_minor_locator(
#         matplotlib.ticker.AutoMinorLocator(5)
#     )

# for num, ax in enumerate(fig.get_axes()):
#     ax.tick_params(axis="both", which="major", labelsize=17)
#     ax.tick_params(axis="both", which="minor", labelsize=17)
#     for axis in ["top", "bottom", "left", "right"]:
#         ax.spines[axis].set_linewidth(1.5)

# fig.set_size_inches(14, 14)
# fig.tight_layout()
# fig.subplots_adjust(wspace=0.1, hspace=0.1)

# plt.savefig("Posteriors_astro_model1.pdf")
