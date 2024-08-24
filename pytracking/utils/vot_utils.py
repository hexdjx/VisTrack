"""
\file vot.py
@brief Python utility functions for VOT integration
@author Luka Cehovin, Alessio Dore
@date 2016
"""

import sys
import copy
import collections
import numpy as np

try:
    import trax
except ImportError:
    raise Exception('TraX support not found. Please add trax module to Python path.')

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])


import numpy as np


def make_full_size(x, output_sz):
    """
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    """
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


def rect_from_mask(mask):
    """
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    """
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def mask_from_rect(rect, output_sz):
    """
    create a binary mask from a given rectangle
    rect: axis-aligned rectangle [x0, y0, width, height]
    output_sz: size of the output [width, height]
    """
    mask = np.zeros((output_sz[1], output_sz[0]), dtype=np.uint8)
    x0 = max(int(round(rect[0])), 0)
    y0 = max(int(round(rect[1])), 0)
    x1 = min(int(round(rect[0] + rect[2])), output_sz[0])
    y1 = min(int(round(rect[1] + rect[3])), output_sz[1])
    mask[y0:y1, x0:x1] = 1
    return mask


def bbox_clip(x1, y1, x2, y2, boundary, min_sz=10):
    """boundary (H,W)"""
    x1_new = max(0, min(x1, boundary[1] - min_sz))
    y1_new = max(0, min(y1, boundary[0] - min_sz))
    x2_new = max(min_sz, min(x2, boundary[1]))
    y2_new = max(min_sz, min(y2, boundary[0]))
    return x1_new, y1_new, x2_new, y2_new


class VOT(object):
    """ Base class for Python VOT integration """

    def __init__(self, region_format, channels=None):
        """ Constructor
        Args:
            region_format: Region format options
        """
        assert (region_format in [trax.Region.RECTANGLE, trax.Region.POLYGON, trax.Region.MASK])

        if channels is None:
            channels = ['color']
        elif channels == 'rgbd':
            channels = ['color', 'depth']
        elif channels == 'rgbt':
            channels = ['color', 'ir']
        elif channels == 'ir':
            channels = ['ir']
        else:
            raise Exception('Illegal configuration {}.'.format(channels))

        self._trax = trax.Server([region_format], [trax.Image.PATH], channels, customMetadata=dict(vot="python"))

        request = self._trax.wait()
        assert (request.type == 'initialize')
        if isinstance(request.region, trax.Polygon):
            self._region = Polygon([Point(x[0], x[1]) for x in request.region])
        elif isinstance(request.region, trax.Mask):
            self._region = request.region.array(True)
        else:
            self._region = Rectangle(*request.region.bounds())
        self._image = [x.path() for k, x in request.image.items()]
        if len(self._image) == 1:
            self._image = self._image[0]

        self._trax.status(request.region)

    def region(self):
        """
        Send configuration message to the client and receive the initialization
        region and the path of the first image
        Returns:
            initialization region
        """

        return self._region

    def report(self, region, confidence=None):
        """
        Report the tracking results to the client
        Arguments:
            region: region for the frame
        """
        assert (isinstance(region, (Rectangle, Polygon, np.ndarray)))
        if isinstance(region, Polygon):
            tregion = trax.Polygon.create([(x.x, x.y) for x in region.points])
        elif isinstance(region, np.ndarray):
            tregion = trax.Mask.create(region)
        else:
            tregion = trax.Rectangle.create(region.x, region.y, region.width, region.height)
        properties = {}
        if not confidence is None:
            properties['confidence'] = confidence
        self._trax.status(tregion, properties)

    def frame(self):
        """
        Get a frame (image path) from client
        Returns:
            absolute path of the image
        """
        if hasattr(self, "_image"):
            image = self._image
            del self._image
            return image

        request = self._trax.wait()

        if request.type == 'frame':
            image = [x.path() for k, x in request.image.items()]
            if len(image) == 1:
                return image[0]
            return image
        else:
            return None

    def quit(self):
        if hasattr(self, '_trax'):
            self._trax.quit()

    def __del__(self):
        self.quit()
