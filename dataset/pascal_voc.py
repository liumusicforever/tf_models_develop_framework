import os


import xml.etree.cElementTree as ET

import logging

logging.basicConfig(level = logging.INFO)

def get_img_list(work_path , limit = None):
    image_list = []
    for root , subdir , files in os.walk(work_path):        
        for img_filename in files:
            if '.jpg' in img_filename :
                image_path = root+'/'+img_filename
                annot_filename = img_filename.replace('.jpg','.xml')
                annot_path = os.path.join(root,'../','Annotations',annot_filename)
                if os.path.exists(image_path) and os.path.exists(annot_path):
                    image_list.append([image_path,annot_path])
            if limit :
                if len(image_list) > limit:
                    return image_list
        
        logging.info('Search on {} , total iamges : {}'.\
                     format(root,len(image_list)))
    return image_list


def xmlreader(xml_file_path):
    boxlist=[]
    if not os.path.exists(xml_file_path):
        logging.warning('No xml file')
        return None
    else:
        with open(xml_file_path) as f:
            root=ET.fromstring(f.read())
        size = root.find('size')

        h = float(size.find('height').text)
        w = float(size.find('width').text)
        
        objects=root.findall('object')
        for ob in objects:
            clss = ob.find('name').text
            bndbox =ob.find('bndbox')
            xmin = max(float(bndbox.find('xmin').text)/w,0)
            ymin = max(float(bndbox.find('ymin').text)/h,0)
            xmax = min(float(bndbox.find('xmax').text)/w,0.9999)
            ymax = min(float(bndbox.find('ymax').text)/h,0.9999)
            label = [clss , xmin , ymin , xmax , ymax]
            boxlist.append(label)
        if len(boxlist)==0:
            boxlist=[]
        return w,h,boxlist
    
def parser(root , limit = None):
    
    datas =  get_img_list(root , limit)
    data_list = []
    for i,(img_path,annot_path) in enumerate(datas):
        w,h,bboxes   = xmlreader(annot_path)
        data_list.append([img_path,bboxes])
        if i % 1000 == 0:
            logging.info('Got training items : {}'.format(len(data_list)))
    logging.info('Finished ! total training items : {}'.format(len(data_list)))
    return data_list    


if __name__ == "__main__":
    data_list = parser('/root/data/VOCdevkit/VOC2012/' , limit = 100)
    print (data_list[0])
    
    
