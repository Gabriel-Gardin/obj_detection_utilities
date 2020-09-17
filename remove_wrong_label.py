import glob
import xml.etree.ElementTree as ET


path_to_remove = "images_1799/"


def remove_wrong_labels():
    for fname in glob.glob(path_to_remove + "*.xml"):
        tree = ET.parse(fname)
        for elem in tree.findall("object/name"):
            if(elem.text == "TFT"):
                print("convertendo")
                elem.text = "TLT"
        tree.write(fname)


if __name__ == "__main__":
    remove_wrong_labels()