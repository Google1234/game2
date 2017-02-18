import config
import os
from datasets.imdb import imdb
import uuid
class fish_imdb(imdb):
    def __init__(self, image_set, data_path,make_image_set=False):
        imdb.__init__(self,data_path)
        self._image_set=image_set
        self._data_path=data_path #data/fish_data/
        self._classes = ('__background__', # always index 0
                         'ALB','BET','DOL','LAG','OTHER','SHARK','YFT')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
	if make_image_set:
	    self.make_image_set()
        self._image_index = self._load_image_set_index()
        assert os.path.exists(self._data_path), \
	    'fish data path does not exist: {}'.format(self._data_path)
	
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',self._image_set,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    def make_image_set(self):
        image_path=os.path.join(self._data_path,"JPEGImages",self._image_set)
        assert os.path.exists(image_path),\
            'JPEGImages Path does not exist: {}'.format(image_path)
        images=[os.path.splitext(file)[0] for file in os.listdir(image_path)]
    
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        file = open(image_set_file,"w")
        file.writelines(["%s\n" % item  for item in images])
        file.close()
