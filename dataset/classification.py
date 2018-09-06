import os
import random

def parser(root , rand = False):
    # get all image and their parent dir name
    image_list = []
    class_table = []
    for clss_idx , clss_name in enumerate(os.listdir(root)):
        class_table.append([str(clss_idx) , str(clss_name)])
        clss_root = os.path.join(root,clss_name)
        if not os.path.isdir(clss_root):continue
        for image_name in os.listdir(clss_root):
            if '.jpg' in image_name:
                image_path = os.path.join(clss_root,image_name)
                if os.path.exists(image_path):
                    image_list.append([image_path,clss_idx])
    if rand:
        random.shuffle(image_list)

    # dump class & id table to local
    with open('classes.txt',"w", encoding='utf-8') as f:
        for row in class_table:
            f.writelines("{}:{}".format(row[0],row[1].encode('utf-8','ignore')) + '\n')
    return image_list