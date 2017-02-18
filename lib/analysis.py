import tools.init_paths
import cPickle
import os
import config

###### code ####################
# according classify results, split images to specific label dir
def split_to_files(images_set):
    path = os.path.join("data", images_set)
    box_file = os.path.join(path, 'boxes.txt')
    with open(box_file, 'rb') as fid:
        all_boxes = cPickle.load(fid)
    score_file = os.path.join(path, 'scores.txt')
    with open(score_file, 'rb') as fid:
        all_scores = cPickle.load(fid)
    class_file = os.path.join(path, 'classes.txt')
    with open(class_file, 'rb') as fid:
        imdb_classes = cPickle.load(fid)
    image_index_file = os.path.join(path, 'image_indexs.txt')
    with open(image_index_file, 'rb') as fid:
        imdb_image_indexs = cPickle.load(fid)
    import pandas as pd
    csv_file=os.path.join(path, 'rst.csv')
    data=pd.read_csv(csv_file)
    for label in config.submit_orders:
        image_indexs=data[data[label]>0]["image"]
        from shutil import copyfile
        images_src_path=os.path.join(config.fish_data_dir,'JPEGImages',images_set)#data/fish_data/submit/JPEGImages
        dest_path=os.path.join(path,label)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for name in image_indexs:
            copyfile(os.path.join(images_src_path,name),os.path.join(dest_path,name))
        with open(os.path.join(path,label+".txt"),"wb") as file:
            file.writelines(["%s\n" % item  for item in image_indexs])
if __name__ == '__main__':
    split_to_files("submit")
