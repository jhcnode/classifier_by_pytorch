from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cla import ClaDetector

detector_factory = {
  'cla': ClaDetector, 
}
