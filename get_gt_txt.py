'''
# get ground truth of testing dataset
'''
import sys
import os
import glob
import xml.etree.ElementTree as ET

'''
# get class
'''
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./val"):
    os.makedirs("./val")
if not os.path.exists("./val/ground-truth"):
    os.makedirs("./val/ground-truth")

for image_id in image_ids:
    with open("./val/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse("VOCdevkit/VOC2007/Annotations/"+image_id+".xml").getroot()
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    difficult_flag = True
            obj_name = obj.find('name').text
            '''
            uncomment the following section if other pre-defined labels 
            are present in annotation file
            '''
            # classes_path = 'model_data/class.txt'
            # class_names = get_classes(classes_path)
            # if obj_name not in class_names:
            #     continue

            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text

            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Conversion completed!")
