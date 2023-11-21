import logging
import os

import torch
import torchvision.io
import xmltodict
from torch import Tensor
from torch.utils.data import Dataset


class PeopleArtDataset(Dataset):
    annotationsSuffixDirPath = "Annotations"
    imagesSuffixDirPath = "JPEGImages"

    def __init__(self, people_art_dir_path, logger_level=logging.INFO):
        self.logger = logging.getLogger("Detection")
        self.logger.setLevel(logger_level)
        self.peopleArtDirPath = people_art_dir_path
        self.annotations = []
        annotations_dir_path = f"{self.peopleArtDirPath}/{self.annotationsSuffixDirPath}"
        self.logger.debug("Start walking")
        for dir_name in next(os.walk(annotations_dir_path))[1]:
            self.logger.debug(f'Found directory: {dir_name}')
            for file_name in next(os.walk(f"{annotations_dir_path}/{dir_name}"))[2]:
                self.logger.debug(f'File name: {file_name}')
                file_path = f"{annotations_dir_path}/{dir_name}/{file_name}"
                with open(file_path) as xml_file:
                    unparsed_str = xml_file.read()
                    xml_dict = xmltodict.parse(unparsed_str)
                    self.logger.debug(f"Parsed dictionary: {xml_dict}")
                    self.logger.debug(f"Folder Name in file: {xml_dict.get('annotation').get('folder')}")
                    self.logger.debug(f"Image File in file: {xml_dict.get('annotation').get('filename')}")
                    self.annotations.append(xml_dict)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index) -> tuple[Tensor, dict[str, Tensor]]:
        current_annotations = self.annotations[index].get("annotation")
        folder_name = current_annotations.get("folder")
        image_name = current_annotations.get("filename")
        image_path = f"{self.peopleArtDirPath}/{self.imagesSuffixDirPath}/{folder_name}/{image_name}"
        image_tensor = torchvision.io.read_image(image_path)
        objects = current_annotations.get("object")

        def get_boxes_labels(current_object):
            bndbox = current_object.get("bndbox")
            boxes = [bndbox.get("xmin"), bndbox.get("ymin"), bndbox.get("xmax"), bndbox.get("ymax")]
            boxes = list(map(float, boxes))
            label = int(current_object.get("name") == "person")
            return {"boxes": boxes, "label": label}

        def update_boxes_labels(boxes_labels, current_boxes_labels):
            boxes_labels["boxes"].append(current_boxes_labels["boxes"])
            boxes_labels["label"].append(current_boxes_labels["label"])

        def boxes_label_to_tensor(boxes_labels):
            return {
                "boxes": torch.Tensor(boxes_labels["boxes"]),
                "label": torch.Tensor(boxes_labels["label"])
            }


        boxes_labels = {"boxes": [], "label": []}
        if isinstance(objects, list):
            # Several objects
            for current_object in objects:
                current_boxes_labels = get_boxes_labels(current_object)
                update_boxes_labels(boxes_labels, current_boxes_labels)
        else:
            # Only one object
            current_boxes_labels = get_boxes_labels(objects)
            update_boxes_labels(boxes_labels, current_boxes_labels)
        boxes_labels_tensor = boxes_label_to_tensor(boxes_labels)
        return image_tensor, boxes_labels_tensor


if __name__ == '__main__':
    logging.basicConfig(filename='output.log', encoding='utf-8', format="%(asctime)s %(name)s [%(levelname)s] %("
                                                                        "message)s")
    dataset = PeopleArtDataset("PeopleArt", logger_level=logging.DEBUG)
    print(dataset[0])
