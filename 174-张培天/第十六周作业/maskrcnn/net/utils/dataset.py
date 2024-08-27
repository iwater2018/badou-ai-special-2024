import numpy as np
import skimage
import logging
import skimage.color
import skimage.io
import skimage.transform

class Dataset(object):
    def __init__(self) -> None:
        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                return
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name
        })
    
    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
    
    def image_reference(self, image_id):
        return ""
    
    def prepare(self,class_map=None):
        def clean_name(name):
            return ",".join(name.split(",")[:1])
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_name = [clean_name(c["name"]) for c in self.class_info]

        self.num_images = len(self.image_info)
        self._image_ids = np.array(self.num_images)
        self.class_from_source_map = {"{}.{}".format(info['source'],info['id']): id for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {'{}.{}'.format(info['source'],info['id']): id for info, id in zip(self.image_info,self._image_ids)}

        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] =[]
            for i, info in enumerate(self.class_info):
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

        def map_source_class_id(slef, source_class_id):
            return self.class_from_source_map[source_class_id]
        
        def get_source_class_id(self,class_id,source):
            info = self.class_info[class_id]
            assert info['source'] == source
            return info['id']
        
        @property
        def image_ids(self):
            return self._image_ids
        
        def source_image_link(self,image_id):
            return self.image_info[image_id]['path']
        
        def load_image(self, image_id):
            image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
            image = image[...,:3]
        return image
    
    def load_mask(self, image_id):
        logging.warning("you are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0,0,0])
        class_ids = np.empty([0],np.int32)
        return mask, class_ids