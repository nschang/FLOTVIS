# ------------------------------
# k-means clustering
# ------------------------------
import glob
import xml.etree.ElementTree as ET

import numpy as np

def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])

def kmeans(box,k):
    # get number of boxes
    row = box.shape[0]
    
    # get location of each point to each box
    distance = np.empty((row,k))
    
    # last cluster location
    last_clu = np.zeros((row,))

    np.random.seed()

    # take 5 random points as cluster centers
    cluster = box[np.random.choice(row,k,replace = False)]
    # cluster = random.sample(row, k)
    while True:
        # get iou of each row from the five points
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)
        
        # get minimum
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        # find median of each class
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data(path):
    data = []
    # find box in each xml annotation
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        if height<=0 or width<=0:
            continue
        
        # get width and height of each object bounding box
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # get width and height
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


if __name__ == '__main__':
    # load xml annotations at './VOCdevkit/VOC2007/Annotations'
    # and generate yolo_anchors.txt
    SIZE = 416
    anchors_num = 9
    # load annotations
    path = r'./VOCdevkit/VOC2007/Annotations'

    # save width and height after conversion to scale
    data = load_data(path)
    
    # k-means clustering
    out = kmeans(data,anchors_num)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out*SIZE)
    data = out*SIZE
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()
