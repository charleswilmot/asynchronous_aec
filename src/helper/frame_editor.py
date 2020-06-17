from PIL import ImageDraw, Image #, ImageFont


class FrameEditor:
    def __init__(self, image):
        self.image = Image.fromarray(image)
        self.drawer = ImageDraw.Draw(self.image)
        self.text_fill = (255, 255, 0)
        self._line_size = 10
        self._column_size = 40

    def _add_text(self, xy, text, fill=None, font=None, anchor=None):
        """
        see https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html#PIL.ImageDraw.PIL.ImageDraw.Draw.text
        """
        self.drawer.text(xy, text, fill=fill, font=font, anchor=anchor)

    def text_object_distance(self, object_distance):
        string = "Object distance (m):\t{: .2f}".formet(object_distance)
        xy = self.to_xy(line=1, column=1)
        self._add_text(xy, string, fill=self.text_fill)

    def text_vergence_error(self, vergence_error):
        string = "Vergence error (deg):\t{: .2f}".formet(vergence_error)
        xy = self.to_xy(line=2, column=1)
        self._add_text(xy, string, fill=self.text_fill)

    def text_episode(self, episode_number, episode_total):
        string = "Episode\t{: 3d}/{: 3d}".formet(episode_number, episode_total)
        xy = self.to_xy(line=3, column=1)
        self._add_text(xy, string, fill=self.text_fill)

    def text_test_case(self, test_case):
        string = "Test case:\t{}".formet(test_case)
        xy = self.to_xy(line=4, column=1)
        self._add_text(xy, string, fill=self.text_fill)

    def to_xy(self, line, column):
        return (line * self._line_size, column * self._column_size)

    def draw_rectangle(self, rectangle):
        self.drawer.rectangle(rectangle, outline=(50, 0, 50, 50))

    def draw_rectangle(self, rectangle):
        for rectangle in rectangles:
            self.draw_rectangle(rectangle)

    def save(self, path, filename):
        self.image.save(path + "/" + filename, "JPEG")

    def overlay_at_center(self, image, alpha=0.5):
        h0 = (self.image.shape[0] - image.shape[0]) // 2
        h1 = h0 + image.shape[0]
        w0 = (self.image.shape[1] - image.shape[1]) // 2
        w1 = w0 + image.shape[1]
        self.image[h0:h1, w0:w1] *= alpha
        self.image[h0:h1, w0:w1] += (1 - alpha) * image


class AnaglyphFrameEditor:
    def __init__(self, image_left, image_right):
        image = np.matmul(
            np.concatenate(
                [left_image * 255, right_image * 255], axis=-1
            ),
            anaglyph_matrix
        ).astype(np.uint8)
        super(self, AnaglyphFrameEditor).__init__(image)
