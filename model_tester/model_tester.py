import numpy as np
import os
import tensorflow as tf
import sys, getopt, glob
from PIL import Image
from object_detection.utils import label_map_util
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


MINIMAL_CONFIDENCE = 0.6

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = "/home/gardin/Documents/Estagio_LET/let_obj_detection/mymodels/ssd_models/SSD_Mobilenetv1_FPN/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "/home/gardin/Documents/Estagio_LET/let_obj_detection/mymodels/ssd_models/SSD_Mobilenetv1_FPN/label_map.pbtxt"

num_classes = 5

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict



def run_inference_test(path_imgs):
    """
    Esta função recebe como parâmetros o caminho para as imagens de teste e retorna um dicionário contendo o nome das imagnes como chaves
    e seus respectivos dados como parâmetros
    """
    image_data_dict = {} #Dicionário que segura o valor de todas as imagens dentro de outros dicionários. 

    detection_graph = tf.Graph()
    #os.chdir(path_imgs)
    TEST_IMAGE_PATHS = glob.glob(path_imgs+"/*.jpg")
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes = num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_count = 0
    for image_path in TEST_IMAGE_PATHS:
        image_name = image_path.split("/")[-1][:-4]
        image_data_dict[image_name] = {} #Adiciona o nome da imagem em questão ao dicionário.
        image_data_dict[image_name]["scores"] = []#cria a lista para salvar os scores.
        image_data_dict[image_name]["boxes"] = []#cria a lista para salvar os boxes.
        image_data_dict[image_name]["classes"] = []#cria a lista para salvar os classes.
        

        image_count += 1
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        
        #Itera por todas as imagnes de teste. 
        counter = 0
        for i in output_dict["detection_scores"]:
            if(i >= MINIMAL_CONFIDENCE):
                image_data_dict[image_name]["scores"].append(i)
                image_data_dict[image_name]["boxes"].append((output_dict["detection_boxes"][counter]))
                image_data_dict[image_name]["classes"].append((output_dict["detection_classes"][counter]))

                """
                scores_sublist.append(i)
                boxes_sublist.append((output_dict["detection_boxes"][counter]))
                classes_sublist.append((output_dict["detection_classes"][counter]))"""
            counter += 1

    return(image_data_dict)


def get_data_from_labels(label_path):
    labels_data_dict = {}
    boxes_xmin = []
    boxes_ymin = []
    boxes_ymax = []
    boxes_xmax = []
    


    LabelPath = glob.glob(label_path+"/*.xml")
    for path in LabelPath:
        label_file_name = path.split("/")[-1][:-4]
        labels_data_dict[label_file_name] = {}
        labels_data_dict[label_file_name]["boxes"] = []
        labels_data_dict[label_file_name]["classes"] = []
        

        #print(label_file_name)
        tree = ET.parse(path)
        for elem in tree.findall("object/name"):
            labels_data_dict[label_file_name]["classes"].append(label_to_num(elem.text))

        for elem in tree.findall("object/bndbox/xmin"):
            boxes_xmin.append(elem.text)

        for elem in tree.findall("object/bndbox/ymin"):
            boxes_ymin.append(elem.text)

        for elem in tree.findall("object/bndbox/ymax"):
            boxes_ymax.append(elem.text)

        for elem in tree.findall("object/bndbox/xmax"):
            boxes_xmax.append(elem.text)

        for i in range(len(boxes_xmax)):
            labels_data_dict[label_file_name]["boxes"].append([boxes_xmin[i], boxes_ymin[i], boxes_xmax[i], boxes_ymax[i]])
        boxes_xmin = []
        boxes_ymin = []
        boxes_ymax = []
        boxes_xmax = []

    return(labels_data_dict)

def label_to_num(label):
    if(label == "BUR"):
        return 1
    elif(label == "REM"):
        return 2
    elif(label == "TLT"):
        return 3
    elif(label == "TRF"):
        return 4
    

def run_test(infereced_data, labeled_data):
    correct_detections = 0
    incorrect_detections = 0
    ignored_detections = 0

    ignored_detections2 = 0

    bur_ignored = 0
    rem_ignored = 0
    tlt_ignored = 0
    trf_ignored = 0

    bur_certo = 0
    rem_certo = 0
    tlt_certo = 0
    trf_certo = 0

    total_bur = count_appearences(labeled_data, 1)
    total_rem = count_appearences(labeled_data, 2)
    total_tlt = count_appearences(labeled_data, 3)
    total_trf = count_appearences(labeled_data, 4)

    for image_name in labeled_data:
        #print(infereced_data[image_name])
        """
        print("-*-*-*-*-*-*-*-*-*-*--*-*-")
        print("infereced --- {}".format(infereced_data[image_name]["classes"]))
        print("labeled --- {}".format(labeled_data[image_name]["classes"]))
        print("-*-*-*-*-*-*-*-*-*-*--*-*-")
        print("\n")
"""
        for label in infereced_data[image_name]["classes"]:
            if(label in labeled_data[image_name]["classes"]):
                correct_detections += 1
                if(label == 1):
                    bur_certo += 1
                elif(label == 2):
                    rem_certo += 1
                elif(label == 3):
                    tlt_certo += 1
                elif(label == 4):
                    trf_certo += 1
            else:
                incorrect_detections += 1

        for labeled_label in labeled_data[image_name]["classes"]:
            if(labeled_label not in infereced_data[image_name]["classes"]):
                ignored_detections2 += 1
                if(labeled_label == 1):
                    bur_ignored += 1
                elif(labeled_label == 2):
                    rem_ignored += 1
                elif(labeled_label == 3):
                    tlt_ignored += 1
                elif(labeled_label == 4):
                    trf_ignored += 1

        """detecçõẽs ignoradas podem ser calculadas a partir do numero de detecções anotadas menos o total de detecções inferidas. 
        Se o numero for negativo o mesmo deve ser igual ao numero de detecções erradas"""
        ignored_detections += len(labeled_data[image_name]["classes"]) - len(infereced_data[image_name]["classes"])
        #if(image_labeled["classes"] in )


    print("\n")
    print("Correct detections = {}".format(correct_detections))
    print("Incorrect detections = {}".format(incorrect_detections))
    print("Ignored detections = {}".format(ignored_detections))

    print("Bur ignorados = {}%".format((bur_ignored/total_bur)*100))
    print("Rem ignorados = {}%".format((rem_ignored/total_rem)*100))
    print("Tlt ignorados = {}%".format((tlt_ignored/total_tlt)*100))
    print("Trf ignorados = {}%".format((trf_ignored/total_trf)*100))

    print("Bur detectados = {}%".format((bur_certo/total_bur)*100))
    print("Rem detectados = {}%".format((rem_certo/total_rem)*100))
    print("Tlt detectados = {}%".format((tlt_certo/total_tlt)*100))
    print("Trf detectados = {}%".format((trf_certo/total_trf)*100))
    
    """
    print("Total trf = {}".format(total_trf))
    print("Trf certo = {}".format(trf_certo))
    print("TLT certo = {}".format(tlt_certo))
    print("rem certo = {}".format(rem_certo))
    print("bur certo = {}".format(bur_certo))
       """ 
def count_appearences(data, desired_class):
    """ Recebe um dicionário labeled_data e a classe que se deseja contar o numero de aparições"""
    total_count = 0
    for image_name in data:
        for classe in data[image_name]["classes"]:
            if(classe == desired_class):
                total_count += 1
    return total_count


def main(argv):
    test_folder = ""
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('detector2_.py -i <data folder>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            test_folder = arg

    infereced_Data = run_inference_test(test_folder)
    labeled_Data = get_data_from_labels(test_folder)
    
    run_test(infereced_Data, labeled_Data)


"""
    print(infereced_Data)
    print("\n")
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    print("\n")
    print(labeled_Data)
"""


if __name__ == '__main__':
    main(sys.argv[1:])