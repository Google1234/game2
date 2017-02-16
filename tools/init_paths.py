"""Set up paths for game."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add classify dir to PYTHONPATH
cly_path = osp.join(this_dir, '..', 'classify')
add_path(cly_path)

# Add config dir to PYTHONPATH
config_path = osp.join(this_dir)
add_path(config_path)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'py-faster-rcnn','caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..','py-faster-rcnn', 'lib')
add_path(lib_path)

