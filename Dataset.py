import logging

import PIL.Image
import torch
import torchvision.io
import xmltodict
from torch import Tensor
from torch.utils.data import Dataset


class PeopleArtDataset(Dataset):
    annotationsSuffixDirPath = "Annotations"
    imagesSuffixDirPath = "JPEGImages"

    def __init__(self, people_art_dir_path, mode="trainval", logger_level=logging.INFO):
        self.logger = logging.getLogger("Detection")
        self.logger.setLevel(logger_level)
        self.peopleArtDirPath = people_art_dir_path
        self.annotations = []
        self.mode = mode
        annotations_dir_path = f"{self.peopleArtDirPath}/{self.annotationsSuffixDirPath}"
        self.logger.debug("Start walking")
        with open(f"{annotations_dir_path}/person_{mode}.txt") as annotations_file:
            for line in annotations_file.readlines():
                img_path, label = line.split()
                self.logger.debug(f"Label: {label}\tImage path: {img_path}")
                annotations = None
                if label == "1":
                    # Have person and annotations file
                    annotations_path = img_path + ".xml"
                    with open(f"{people_art_dir_path}/{self.annotationsSuffixDirPath}/{annotations_path}") as xml_file:
                        unparsed_str = xml_file.read()
                        annotations = xmltodict.parse(unparsed_str)
                else:
                    # Don't have person and annotations file
                    with PIL.Image.open(f"{people_art_dir_path}/{self.imagesSuffixDirPath}/{img_path}") as img_file:
                        style_dir, img_name = img_path.split("/")
                        w, h = img_file.size
                        annotations = {"annotation": {"filename": img_name, "folder": style_dir,
                                                      "object": {"name": "unknown",
                                                                 "bndbox": {"xmin": 0, "ymin": 0, "xmax": w,
                                                                            "ymax": h}}}}

                self.logger.debug(f"Parsed annotations: {annotations}")
                self.annotations.append(annotations)

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
            label = 1 if current_object.get("name") == "person" else -1
            return {"boxes": boxes, "labels": label}

        def update_boxes_labels(boxes_labels, current_boxes_labels):
            boxes_labels["boxes"].append(current_boxes_labels["boxes"])
            boxes_labels["labels"].append(current_boxes_labels["labels"])

        def boxes_label_to_tensor(boxes_labels):
            return {
                "boxes": torch.as_tensor(boxes_labels["boxes"]),
                "labels": torch.as_tensor(boxes_labels["labels"])
            }

        boxes_labels = {"boxes": [], "labels": []}
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
