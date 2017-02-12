import tools.init_paths
import config
import caffe
from datasets.factory import get_imdb
import cv2

relative_path="py-faster-rcnn"
#######################################
#load model#
gpu_id=0
prototxt=relative_path+"/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt"
caffemodel=relative_path+"/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_92264.caffemodel"
imdb_name="voc_2007_test"

caffe.set_mode_gpu()
caffe.set_device(gpu_id)
net = caffe.Net(prototxt, caffemodel, caffe.TEST)
imdb = get_imdb(imdb_name)
#######################################
#detect each image#

import numpy as np
from nms.gpu_nms import gpu_nms
import os

PIXEL_MEANS=np.array([[[102.9801, 115.9465, 122.7717]]]) #__C.PIXEL_MEANS
SCALES=(600,) #__C.TEST.SCALES = (600,)
MAX_SIZE=1000 #__C.TEST.MAX_SIZE = 1000
thresh=0.05 #pre score low than this value was discard
NMS=0.3 #__C.TEST.NMS = 0.3
det_image_save_path=config.data_path+"det_result/"

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -=PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []
    for target_size in SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > MAX_SIZE:
            im_scale = float(MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
    # Create a blob to hold the input images

    #blob = im_list_to_blob(processed_ims)
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in processed_ims]).max(axis=0)
    num_images = len(processed_ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = processed_ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    return blob, np.array(im_scale_factors)
def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
def vis_save_detections(im, _image_index,class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            #plt.show()

    if os.path.exists(det_image_save_path)==False:
        os.mkdir(det_image_save_path)
    plt.savefig("demo_result/"+_image_index+"jpg")

num_images = len(imdb.image_index)
# all detections are collected into:
#    all_boxes[cls][image] = N x 5 array of detections in
#    (x1, y1, x2, y2, score)
all_boxes = [[[] for _ in xrange(num_images)]
             for _ in xrange(imdb.num_classes)]

for i in xrange(num_images):
    im = cv2.imread(imdb.image_path_at(i))

    #create input blobs
    blobs = {'data': None, 'rois': None}
    blobs['data'], im_scales = _get_image_blob(im)

    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)
    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    #get network output
    assert len(im_scales) == 1, "Only single-image batch implemented"
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scales[0]

    scores = blobs_out['cls_prob']
    # Apply bounding-box regression deltas
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)
    #return scores, pred_boxes

    #process network output
    # skip j = 0, because it's the background class
    for j in xrange(1, imdb.num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = gpu_nms(cls_dets, NMS, gpu_id)
        cls_dets = cls_dets[keep, :]

        vis_save_detections(im, imdb._image_index[j],imdb.classes[j], cls_dets)#plot detection and save image
        all_boxes[j][i] = cls_dets
