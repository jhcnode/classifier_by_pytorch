from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cla import CLATrainer


train_factory = {
  'cla': CLATrainer, 
}
