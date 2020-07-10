import matplotlib.pyplot as plt


def plot_heatmap(heatmap_array, filename, path):
    """
    :param path:
    :param array:
    :param ratio:
    :return:
    """
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    print(heatmap_array)
    print(heatmap_array.shape)
    if len(heatmap_array.shape) == 2:
        plt.imshow(heatmap_array)
    else:
        plt.imshow(heatmap_array[0, ...], interpolation='gaussian')
    plt.colorbar(orientation='vertical')
    plt.savefig(path + '/' + filename + '.png')
    plt.close(fig)

