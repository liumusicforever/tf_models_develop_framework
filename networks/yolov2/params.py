
batch_size = 30
save_summary_steps = 50
num_epochs = 160
pre_trained = ''

# network parameter
network_params = {
    "lr" : 0.0001
}

# For multiple scale training
training_scale = [288,352,416,544,608]

# For yolov2 loss
object_scale = 5
no_object_scale = 1
class_scale = 1
coordinates_scale = 1


# class name to class id
classes_mapping = {
    'person':0,
    'bicycle':1,
    'car':2,
    'motorcycle':3,
    'airplane':4,
    'bus':5,
    'train':6,
    'truck':7,
    'boat':8,
    'traffic light':9,
    'fire hydrant':10,
    'stop sign':11,
    'parking meter':12,
    'bench':13,
    'bird':14,
    'cat':15,
    'dog':16,
    'horse':17,
    'sheep':18,
    'cow':19,
    'elephant':20,
    'bear':21,
    'zebra':22,
    'giraffe':23,
    'backpack':24,
    'umbrella':25,
    'handbag':26,
    'tie':27,
    'suitcase':28,
    'frisbee':29,
    'skis':30,
    'snowboard':31,
    'sports ball':32,
    'kite':33,
    'baseball bat':34,
    'baseball glove':35,
    'skateboard':36,
    'surfboard':37,
    'tennis racket':38,
    'bottle':39,
    'wine glass':40,
    'cup':41,
    'fork':42,
    'knife':43,
    'spoon':44,
    'bowl':45,
    'banana':46,
    'apple':47,
    'sandwich':48,
    'orange':49,
    'broccoli':50,
    'carrot':51,
    'hot dog':52,
    'pizza':53,
    'donut':54,
    'cake':55,
    'chair':56,
    'couch':57,
    'potted plant':58,
    'bed':59,
    'dining table':60,
    'toilet':61,
    'tv':62,
    'laptop':63,
    'mouse':64,
    'remote':65,
    'keyboard':66,
    'cell phone':67,
    'microwave':68,
    'oven':69,
    'toaster':70,
    'sink':71,
    'refrigerator':72,
    'book':73,
    'clock':74,
    'vase':75,
    'scissors':76,
    'teddy bear':77,
    'hair drier':78,
    'toothbrush':79,
}
# classes_mapping = {
#     'aeroplane' : 0,
#     'bicycle' : 1,
#     'bird' : 2,
#     'boat' : 3,
#     'bottle' : 4,
#     'bus' : 5,
#     'car' : 6,
#     'cat' : 7,
#     'chair' : 8,
#     'cow' : 9,
#     'diningtable' : 10,
#     'dog' : 11,
#     'horse' : 12,
#     'motorbike' : 13,
#     'person' : 14,
#     'pottedplant' : 15,
#     'sheep' : 16,
#     'sofa' : 17,
#     'train' : 18,
#     'tvmonitor' : 19
# }


# classes_mapping = {
#     '1' : 0,
#     'canDrink' : 1,
#     'bottleDrink' : 2,
#     'foil' : 3,
#     'boxCookie' : 4,
# }

num_classes = len(classes_mapping)


# voc anchors
# anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
anchors = [
    [0.01644737, 0.02138158],
    [0.02631579, 0.04934211],
    [0.05427632, 0.03782895],
    [0.04934211, 0.10032895],
    [0.10197368, 0.07401316],
    [0.09703947, 0.19572368],
    [0.19078947, 0.14802632],
    [0.25657895, 0.32565789],
    [0.61348684, 0.53618421]]
num_anchors = len(anchors)
