import cv2
# from matplotlib import pyplot as plt
import config


class Screen(config.GameConfig):
    def __init__(self, screen):
        self.screen = screen

        # Rendering
        if self.do_render:
            cv2.startWindowThread()
            cv2.namedWindow(winname=self.environment)

    def copy(self):
        new_obj = Screen(screen=self.screen)
        return new_obj

    def to_colour(self, cv2COLOR, inplace=False):
        if inplace:
            new_obj = self
        else:
            new_obj = self.copy()
        new_obj.screen = cv2.cvtColor(new_obj.screen, cv2COLOR)
        return new_obj

    def to_rgb(self, inplace=False):
        return self.to_colour(cv2COLOR=cv2.COLOR_BGR2RGB, inplace=inplace)

    def to_grey(self, inplace=False):
        return self.to_colour(cv2COLOR=cv2.COLOR_BGR2GRAY, inplace=inplace)

    def to_binary(self, inplace=False):
        new_obj = self.to_grey(inplace=inplace)
        result, new_obj.screen = cv2.threshold(
            src=new_obj.screen,
            thresh=1, maxval=255,
            type=cv2.THRESH_BINARY)
        new_obj.screen = new_obj.screen / 255.
        return new_obj

    def render(self):
        cv2.imshow(winname=self.environment,
                   mat=self.to_rgb().screen)

    def resize(self, inplace=False):
        if inplace:
            new_obj = self
        else:
            new_obj = self.copy()
        new_obj.screen = cv2.resize(new_obj.screen, new_obj.resize_shape)
        return new_obj

    # def render_grid(self):
    #     plt.subplot(231), plt.imshow(
    #         self.screen, 'gray'), plt.title('RGB')
    #
    #     plt.subplot(232), plt.imshow(
    #         self.screen_bgr(), 'gray'), plt.title('BGR')
    #
    #     plt.subplot(233), plt.imshow(
    #         self.screen_grey(), 'gray'), plt.title('Grey')
    #
    #     plt.subplot(234), plt.imshow(
    #         self.screen_binary(), 'gray'), plt.title('Binary')
