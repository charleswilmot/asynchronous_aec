from PIL import ImageDraw, Image, ImageFilter #, ImageFont
import numpy as np
import matplotlib.pyplot as plt

anaglyph_matrix = np.array([
    [0.299, 0    , 0    ],
    [0.587, 0    , 0    ],
    [0.114, 0    , 0    ],
    [0    , 0.299, 0.299],
    [0    , 0.587, 0.587],
    [0    , 0.114, 0.114],
    ])


def overlay_at_center(image, overlay, alpha=0.4):
    """
    Overlays 'overlay' over center of image
    :param image: image as numpy array
    :param overlay: overlay as numpy array
    :param alpha: transparency value of the image (between 0.0 and 1.0)
    :return: merged images as numpy array
    """
    image = image.astype(np.float64)
    overlay = overlay.astype(np.float64)
    h0 = (image.shape[0] - overlay.shape[0]) // 2
    h1 = h0 + overlay.shape[0]
    w0 = (image.shape[1] - overlay.shape[1]) // 2
    w1 = w0 + overlay.shape[1]
    image[h0:h1, w0:w1] *= alpha
    image[h0:h1, w0:w1] += (1 - alpha) * overlay
    return image.astype(np.uint8)


def normalize_heatmap(heatmap):
    """
    Normalize the array to values between 0 and 1
    :param heatmap: 2D - heatmap as numpy array
    :return: 2D - heatmap as numpy array with values between 0 and 1
    """
    xmax, xmin = np.amax(heatmap), np.amin(heatmap)
    heatmap = (heatmap - xmin) / (xmax - xmin)
    return heatmap


def color_map_heatmap(heatmap, colormap='hot'):
    """
    Transform (normalized) array with help of color map
    :param heatmap: (normalized) 2D - heatmap as numpy array
    :param colormap: matplotlib color map
    :return: 3D - rgba numpy array
    """
    cm = plt.get_cmap(colormap)
    heatmap = cm(heatmap)
    return heatmap


def get_image_subset(image, filter_size, stride):
    """
    Returns the conv subset images
    :param image:
    :param filter_size:
    :param stride:
    :return:
    """
    rows, cols, _ = image.shape
    patches = []
    for row in np.arange(0, rows - filter_size + 1, stride):
        patches_row = []
        for col in np.arange(0, cols - filter_size + 1, stride):
            patches_row.append(np.pad(image[row:row + filter_size, col:col + filter_size], ((1, 1), (1, 1), (0, 0)),
                                      'constant'))
        patches.append(patches_row)
    return patches


def cropND(image, bounding):
    """
    Extracts center crop of numpy image array
    :param image:
    :param bounding:
    :return:
    """
    import operator
    start = tuple(map(lambda a, da: a // 2 - da // 2, image.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return image[slices]


def generate_heatmap_pil(heatmap):
    """
    Takes numpy array and return heatmap as pil image
    :param heatmap:
    :return:
    """
    heatmap = normalize_heatmap(heatmap)
    heatmap = color_map_heatmap(heatmap)
    return Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8))


class FrameEditor:
    def __init__(self, path, image):
        self._path = path
        self.text_fill = (255, 255, 0)
        self.line_size = 12
        self.column_size = 15
        self.image = Image.fromarray(image)
        self.drawer = ImageDraw.Draw(self.image)

    def _add_text(self, xy, text, fill=None, font=None, anchor=None):
        """
        see https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html#PIL.ImageDraw.PIL.ImageDraw.Draw.text
        """
        self.drawer.text(xy, text, fill=fill, font=font, anchor=anchor)

    def text_episode(self, episode_number, episode_total):
        string = "Episode\t{: 3d}/{: 3d}".format(episode_number, episode_total)
        xy = self.to_xy(line=0.5, column=1)
        self._add_text(xy, string, fill=self.text_fill)

    def text_object_distance(self, object_distance):
        string = "Object distance (m):\t{: .2f}".format(object_distance)
        xy = self.to_xy(line=2, column=1)
        self._add_text(xy, string, fill=self.text_fill)

    def text_vergence_error(self, vergence_error):
        string = "Vergence error (deg):\t{: .2f}".format(vergence_error)
        xy = self.to_xy(line=3, column=1)
        self._add_text(xy, string, fill=self.text_fill)

    def text_speed_error(self, speed_error):
        string = "Delta speed error (deg):\t{: .2f}".format(speed_error)
        xy = self.to_xy(line=4, column=1)
        self._add_text(xy, string, fill=self.text_fill)

    def text_test_case(self, test_case):
        string = "Test case:\t{}".format(test_case)
        xy = self.to_xy(line=18, column=1)
        self._add_text(xy, string, fill=self.text_fill)

    def to_xy(self, line, column):
        return int(column * self.column_size), int(line * self.line_size)

    def draw_rectangle(self, rectangle):
        self.drawer.rectangle(rectangle, outline=(50, 0, 50, 50))

    def draw_rectangles(self, rectangles):
        for rectangle in rectangles:
            self.draw_rectangle(rectangle)

    def save(self, filename, image=None, target_image_size=512, height_or_width=""):
        if not image:
            image = self.image
        width, height = image.size
        if height_or_width == 'both':
            scale_factor = max(1, min(target_image_size/width, target_image_size/height))
        elif height_or_width == 'height':
            scale_factor = max(1, target_image_size / height)
        elif height_or_width == 'width':
            scale_factor = max(1, target_image_size / width)
        else:
            scale_factor = 1
        image = image.resize(size=(int(width*scale_factor), int(height*scale_factor)), resample=Image.NEAREST)  # Image.BOX
        image.save(self._path + "/" + filename + ".jpg", "JPEG")

    def add_video_frame_info(self, rectangles=None, object_distance=None, speed_error=None, vergence_error=None,
                             episode=None, max_episode='-', test_case=None):
        if rectangles is not None:
            self.draw_rectangles(rectangles)
        if episode is not None:
            self.text_episode(episode, max_episode)
        if object_distance is not None:
            self.text_object_distance(object_distance)
        if vergence_error is not None:
            self.text_vergence_error(vergence_error)
        if speed_error is not None:
            self.text_speed_error(speed_error)
        if test_case is not None:
            self.text_test_case(test_case)


class AnaglyphFrameEditor(FrameEditor):
    def __init__(self, path, left_image, right_image):
        image = np.matmul(
            np.concatenate(
                [left_image * 255, right_image * 255], axis=-1
            ),
            anaglyph_matrix
        ).astype(np.uint8)
        self.left_image = Image.fromarray((left_image * 255).astype(np.uint8))
        self.right_image = Image.fromarray((right_image * 255).astype(np.uint8))
        super(AnaglyphFrameEditor, self).__init__(path, image)


class SaliencyFrameEditor(AnaglyphFrameEditor):
    def __init__(self, path, left_image, right_image, heatmap,):
        self.heatmap = heatmap[0, ...]  # Heatmap as unprocessed numpy array
        self.heatmap_pil = generate_heatmap_pil(self.heatmap)
        super(SaliencyFrameEditor, self).__init__(path, left_image, right_image)

    # Todo: This method will be refactored to use a conv method for patch creation
    def image_patches(self, scale_size, ratio, filter_size, stride, filename):
        heatmap_scale_width = int(((scale_size[1] - filter_size + stride) / stride) * (filter_size + 2))
        heatmap_scale_height = int(((scale_size[0] - filter_size + stride) / stride) * (filter_size + 2))
        heatmap_pil = self.heatmap_pil.resize(size=(heatmap_scale_width, heatmap_scale_height), resample=Image.NEAREST)  # Image.BOX

        crop_size = scale_size * ratio
        frame_crop = cropND(np.array(self.image), (heatmap_scale_height, heatmap_scale_width))
        frame_crop_pil = Image.fromarray(frame_crop)
        frame_crop_downscale = frame_crop_pil.resize(size=(scale_size[1], scale_size[0]), resample=Image.NEAREST)
        frame_crop = np.array(frame_crop_downscale)
        patches = get_image_subset(frame_crop, filter_size, stride)

        for pos, x in enumerate(patches):
            image = x[0]
            for pos2, patch in enumerate(x[1:]):
                image = np.hstack([image, patch])
            if pos != 0:
                combines_images = np.vstack([combines_images, image])
            else:
                combines_images = image

        combined = np.hstack([combines_images, np.array(heatmap_pil)])
        combined_pil = Image.fromarray(combined)
        combined_pil.save(self._path + "/" + filename + ".jpeg", "JPEG", height_or_width='height')

    def overlay_heatmap(self, name, heatmap_target_size):
        heatmap = self.heatmap_pil.resize(size=heatmap_target_size, resample=Image.NEAREST)  # Alternative: Image.BOX
        image_heatmap_overlay = overlay_at_center(np.array(self.image), np.array(heatmap))
        image_heatmap_overlay = Image.fromarray(image_heatmap_overlay)
        self.save(name, image_heatmap_overlay)
        return np.array(image_heatmap_overlay)

    def add_saliency_frame_info(self, test_case):
        if test_case is not None:
            self.text_test_case(test_case)