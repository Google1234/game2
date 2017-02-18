import cPickle
import numpy as np
import tools.init_paths
import test_net
import config
import cPickle
import os

######  Variable ############
retest_flag=True
thresh=0.5
images_set="submit"
path = os.path.join("data", images_set)
submit_orders=config.submit_orders


######  Code  ###############
if retest_flag:
    #####################
    # use net/test_net 
    all_boxes, \
    all_scores, \
    imdb_classes, \
    imdb_image_indexs = test_net.test(gpu_id=0,
                                      prototxt="py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt",
                                      caffemodel="py-faster-rcnn/output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_92264.caffemodel",
                                      thresh=0.05,
                                      NMS=0.3,
                                      images_set=images_set,
                                      data_dir=config.fish_data_dir,
                                      make_image_set=True)
    ######################
    #use cpickle ,write to file 
    if not os.path.exists(path):
        os.makedirs(path)
    box_file = os.path.join(path, 'boxes.txt')
    with open(box_file, 'wb') as fid:
        cPickle.dump(all_boxes, fid, cPickle.HIGHEST_PROTOCOL)
    score_file = os.path.join(path, 'scores.txt')
    with open(score_file, 'wb') as fid:
        cPickle.dump(all_scores, fid, cPickle.HIGHEST_PROTOCOL)
    class_file = os.path.join(path, 'classes.txt')
    with open(class_file, 'wb') as fid:
        cPickle.dump(imdb_classes, fid, cPickle.HIGHEST_PROTOCOL)
    image_index_file = os.path.join(path, 'image_indexs.txt')
    with open(image_index_file, 'wb') as fid:
        cPickle.dump(imdb_image_indexs, fid, cPickle.HIGHEST_PROTOCOL)
else: 
    box_file = os.path.join(path, 'boxes.txt')
    with open(box_file,'rb') as fid:
        all_boxes=cPickle.load(fid)
        
    score_file = os.path.join(path, 'scores.txt')
    with open(score_file,'rb') as fid:
        all_scores=cPickle.load(fid)

    class_file = os.path.join(path, 'classes.txt')
    with open(class_file,'rb') as fid:
        imdb_classes=cPickle.load(fid)

    image_index_file = os.path.join(path, 'image_indexs.txt')
    with open(image_index_file,'rb') as fid:
        imdb_image_indexs=cPickle.load(fid)

##img_0010  --> img_0010.jpg
image_indexs = map(lambda e:  e + ".jpg", imdb_image_indexs)

rst=np.zeros((len(image_indexs),len(submit_orders)))
max_scores=[0 for _ in xrange(len(imdb_classes))]
for index in range(len(image_indexs)):
    score=all_scores[index]
    for j in xrange(len(imdb_classes)):
        if len(score[j])==0:
            max_scores[j]=0
        else:
            max_scores[j]=max(score[j])
    label_index=max_scores.index(max(max_scores))
    if max_scores[label_index]<thresh:
        label="NoF"
    else:
        label=imdb_classes[label_index]
    index_orders=submit_orders.index(label)
    rst[index][index_orders]=1
    
#############################################
# write to file 
import pandas as pd
data=pd.DataFrame(rst,index=image_indexs,columns=submit_orders)
#print data
data.to_csv(path+"/rst.csv",index_label="image")

