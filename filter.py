import os
import xml.etree.ElementTree as ET
import cv2
import shutil

xml_path = "/data/shw/zhixing/ZH001/Annotations"


def load_pascal_annotation(xml):

    filename = os.path.join(xml_path, xml)
    tree = ET.parse(filename)
    objs = tree.findall('object')

    if(len(objs)==0):
        print(xml)
        os.remove(filename)

if __name__ == "__main__":

    xmls = os.listdir("/data/shw/zhixing/ZH001/Annotations")
    for x in xmls:
        load_pascal_annotation(x)