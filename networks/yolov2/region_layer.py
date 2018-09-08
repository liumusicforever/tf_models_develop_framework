
import numpy as np

import networks.yolov2.params as params

def lastlayer2detection(lastlayer , thresh = 0.5):
    '''
    Parameter
        lastlayer :
            dtype : np.array
            shape : (batch_size , H , W , num_anchors , (5(x , y , w , h , prob) + num_clss))
    Return
        detection : 
            dtype : list
            shape : (batch_size , num_bboxes , 5 (clss,x1 , y1 , x2 , y2 ))
    '''
    batch_size = lastlayer.shape[0]
    prob_map = lastlayer[...,4:5]
    res_idx = np.where( prob_map >= thresh)
    # res_idx : batch_idx , h_idx , w_idx , anchor_idx , ...
    
    # Compute cx & cy
    Cxy = np.stack((res_idx[1] ,res_idx[2]), axis=-1)
    # Cxy : (num_boxes , cx cy)

    
    # Compute pw & ph
    anchor_idx = res_idx[3]
    Pwh = np.array(params.anchors)[(np.array(anchor_idx))]
    # Pwh : (num_boxes , pw ph)
    
    pred_boxes = lastlayer[res_idx[:4]]
    # pred_boxes : (num_boxes , 4(tx,ty,tw,th,...))
    xycen = (pred_boxes[...,0:2] + Cxy) / lastlayer.shape[1:3]
    xywh = (pred_boxes[...,2:4] * Pwh) 

    xymin = xycen - (xywh/2)
    xymax = xycen + (xywh/2)
    

    cord= np.stack((xymin , xymax ), axis=1).reshape(-1,4).tolist()
    clss = np.argmax(pred_boxes[...,5:], -1).tolist()
    # cord : (num_boxes , 4)
    # clss : (num_boxes , 1)
    
    batch_idxs , batch_num_boxes = np.unique(res_idx[0], return_counts=True)
    batch_idxs = batch_idxs.tolist()
    batch_num_boxes = batch_num_boxes.tolist()
    # batch_idxs : [batch_id 1, batch id 2 ,...]
    # batch_num_boxes : [num_boxes of batch_id 1 , num_boxes of batch_id 2,..]
    
    detection = []
    box_id = 0
    for batch_idx in range(batch_size):
        batch_prediction = []
        if batch_idx in batch_idxs:
            for i in range(batch_num_boxes[batch_idxs.index(batch_idx)]):                 
                batch_prediction.append([clss[box_id]] + cord[box_id])
                box_id += 1
        detection.append(batch_prediction)
    
    return detection

def detection2lastlayer(detection , out_shapes = (4,4)):
    '''
    Parameter
        detection : 
            dtype : np.array
            shape : (batch_size , num_bboxes , 5 (clss , x1 , y1 , x2 , y2))
        out_shapes:
            dtype : tuple
            shape : (W , H)
    Return
        lastlater : 
            dtype : np.array
            shape : (batch_size , H , W , num_anchors , (5(x , y , w , h , prob) + num_clss))
    '''
    out_shapes = np.array(out_shapes)
    batch_size = detection.shape[0]
    
    
    # Compute which grid is center of boxes belong to 

    boxes_cen = (detection[...,3:5] + detection[...,1:3]) / 2
    boxes_grid_idx = (boxes_cen / (1.0 / out_shapes)).astype(np.int8).astype(np.float)

    # boxes_grid_idx : (batch_size , num_boxes , w and h grid id)
    boxes_xy_offset = (boxes_cen - (boxes_grid_idx/out_shapes))*out_shapes
    # boxes_xy_offset : (batch_size , num_boxes , offset x y)
    
    # Compute one hot encoding of each boxes
    
    boxes_clss = detection[...,0:1]    
    boxes_clss_one_hot = (np.arange(params.num_classes) == boxes_clss[...,None]).astype(int)
    boxes_clss_one_hot = boxes_clss_one_hot.reshape(batch_size , -1 , params.num_classes)
    # boxes_clss_one_hot : (batch_size , num_boxes , num_classes )
    

    
    # Compute which anchors is boxes belong to
    
    boxes_wh = detection[...,3:5] - detection[...,1:3]
    # to avoid negative value of h,w
    boxes_wh = np.where(boxes_wh < 0 , 0 , boxes_wh)
    anchors_wh = np.array(params.anchors)
    # boxes_wh : (batch_size , num_boxes , wh)
    # anchors_wh : (num_anchor , wh)
    boxes_wh = boxes_wh[:,:, np.newaxis ,...]
    anchors_wh = anchors_wh[np.newaxis , np.newaxis , : , ...]
    # (batch_size , num_boxes, num_anchor , wh)

    # Compute boxes and anchor region
    
    boxes_region = boxes_wh[...,0:1] * boxes_wh[...,1:2]
    anchors_region = anchors_wh[...,0:1] * anchors_wh[...,1:2]
    
    # Compute boxes and anchor intersection
    inter_w = np.minimum(boxes_wh[...,0:1] , anchors_wh[...,0:1])
    inter_h = np.minimum(boxes_wh[...,1:2] , anchors_wh[...,1:2])
    inter_region = inter_w * inter_h
    # inter_region : (batch_size, num_boxes , num_anchor, region_size)
    iou = inter_region / (boxes_region + anchors_region - inter_region)
    
    best_iou = np.argmax(iou, axis=2)

    # best_iou : (batch_size , num_boxes , which anchor is most close to detection)
    
    # Define final output space 
    lastlayer = np.zeros((batch_size ,out_shapes[0] , out_shapes[1] , params.num_anchors ,
                          5 + params.num_classes))
    # lastlaer : (batch_size , H , W , num_anchor , 5*num_classes )
    
    # Insert each boxes information to output
    
    bgi_dim = boxes_grid_idx.shape
    # bgi_dim : (batch_size , num_boxes , w and h grid id)
    for batch_idx in range(bgi_dim[0]):
        for box_idx in range(bgi_dim[1]):
            # region size of box
            box_region = boxes_region[batch_idx ,box_idx , 0 , 0]
            # skip padding blank boxes
            if not box_region > 10e-9:
                continue
                
            cols, rows = [int(i) for i in boxes_grid_idx[batch_idx , box_idx ,:2]]
            anchor_idx = best_iou[batch_idx , box_idx , 0]
            
            
            # compute box offset of w , h 
            box_wh = boxes_wh[batch_idx,box_idx,:,:]
            anch_wh = anchors_wh[0,:,anchor_idx,:]
            box_wh_offset = (box_wh / anch_wh).reshape(-1)
            
            # box offset of x , y 
            box_xy_offset = boxes_xy_offset[batch_idx , box_idx , :]
            
            # box one hot of classes 
            box_clss_one_hot = boxes_clss_one_hot[batch_idx , box_idx , :]
            
            
            
            # merge [x , y , w , h , prob , clss1 , clss2 ...]
            merge_arrs = (box_xy_offset ,box_wh_offset , [1], box_clss_one_hot)
            pred = np.concatenate(merge_arrs, axis=0)
            lastlayer[batch_idx , cols , rows , anchor_idx , :] = pred
    
            
    return lastlayer