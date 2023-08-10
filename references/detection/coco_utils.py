import os

import torch
import torch.utils.data
import torchvision
import transforms as T
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    # FIXME: This is... awful?
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(root, image_set, transforms, mode="instances", use_v2=False, with_masks=False):
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    if use_v2:
        from torchvision.datasets import wrap_dataset_for_transforms_v2

        dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
        target_keys = ["boxes", "labels", "image_id"]
        if with_masks:
            target_keys += ["masks"]
        dataset = wrap_dataset_for_transforms_v2(dataset, target_keys=target_keys)
    else:
        # TODO: handle with_masks for V1?
        t = [ConvertCocoPolysToMask()]
        if transforms is not None:
            t.append(transforms)
        transforms = T.Compose(t)

        dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset

class CostumCocoDataset():
    """If you create your own annotaion from different site or application
    and have it in json format you can use this class to export 
    segmented images, segmented images with bboxes, masks and 
    output file for yolo model in .txt format

    first of all you need create instance with:
    `annotation_path`: path to you annotation directory
    `image_dir`: path to you image directory

    after creat instance of class call `display_image()`
    
    image_id='random'    if need specific image mention image_id in json file
                        if you have many images with json file loop throw them all
                        and then save them in folder for train/val/test
                        save masks in numpy format for better exprience
    show_polys=True    show image with annotation
    show_mask=True    show mask in grayscale
    show_bbox=True    show images with segments and bboxes
    show_crowds=True
    yolo_txt=True --> save txt file in "/content/dataset/labels"
    
    
    """
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.colors = [(255, 0, 255), (255, 255, 0), (255, 100, 100),
                        (0, 255, 255), (100, 100, 200), (255, 0, 0),
                        (0, 255, 0), (0, 0, 255), (180, 100, 120)]

        json_file = open(self.annotation_path)
        self.coco = json.load(json_file)
        json_file.close()

        self.process_info()
        self.process_images()
        self.process_segmentations()


    def display_info(self):
        print('Dataset Info:')
        print('=============')
        for key, item in self.info.items():
            print(f'  {key}: {item}')

    def display_image(self, image_id='random', show_polys=True,
                      show_mask=True, show_bbox=True, show_crowds=True, yolo_txt=True):
        print('Image:')
        print('======')
        if image_id == 'random':
            image_id = random.choice(list(self.images.keys()))

        # Print the image info
        image_info = self.images[image_id]
        for key, val in image_info.items():
            print(f'  {key}: {val}')

        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate the size and adjusted display size
        image_width, image_height = image.shape[0], image.shape[1]
        adjusted_width = 128
        adjusted_ratio = 1
        adjusted_height = int(adjusted_ratio * adjusted_width)


        # Create list of polygons to be drawn
        polygons = {}
        bbox_polygons = {}
        rle_regions = {}
        poly_colors = {}
        for i, segm in enumerate(self.segmentations[image_id]):
            polygons_list = []

            # Add the polygon segmentation
            for segmentation_points in segm['segmentation']:
                polygons_list.append(np.array(segmentation_points).astype(int))

            polygons[segm['id']] = polygons_list

            # extract bbox as int for use in opencv
            bbox = [int(x) for x in segm['bbox']]
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
            bbox_polygons[segm['id']] = bbox

            # extract point for use in opencv
            points= []
            for seg_id, points_list in polygons.items():

                pnts = []
                idx = []
                for item in points_list:
                    idx.append(len(item))
                    i = 0
                    while i < (len(item) - 1):
                        pnts.append(item[i:i+2].tolist())
                        i += 2
                    points.append(pnts)

            yolo_points = points_list

            # segment the orginal image
            overlay = image.copy()
            for i in range(len(polygons)):
                pts = np.array(points[i]).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], self.colors[i])

            alpha = 0.4
            segmented_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            segmented_image_resize = cv2.resize(segmented_image, (adjusted_width, adjusted_height))

            # extract the mask of segmented images
            mask = np.zeros((image_width, image_height ), dtype=np.uint8)
            for i in range(len(polygons)):
                pts = np.array(points[i]).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], (i+1,0, 0))

            mask = cv2.resize(mask, (adjusted_width, adjusted_height))

        if show_polys:
            plt.imshow(segmented_image_resize)
            plt.show()

        if show_mask:
            plt.imshow(mask)
            plt.show()

        if show_bbox:
            for i, (seg_id, points) in enumerate(bbox_polygons.items()):
                pnt1 = (points[0], points[1])
                pnt2 = points[2], points[3]
                img_with_bbox = cv2.rectangle(segmented_image, pnt1, pnt2, self.colors[i], 3)
            img_with_bbox = cv2.resize(img_with_bbox, (adjusted_width, adjusted_height))
            plt.imshow(img_with_bbox)
            plt.show()

        # extract point items as txt file for yolo8 segmentation
        if yolo_txt:
            yolo_ann_path = "/content/dataset/labels"
            os.makedirs(yolo_ann_path)
            file_name = image_info['file_name'][:-4]
            for item in yolo_points:
                with open(f"{yolo_ann_path}/{file_name}.txt", 'a') as f:
                        f.write("\n0 ")
                        for i, data in enumerate(item):
                            if i % 2 == 0:
                                data /= image_height
                            else:
                                data/= image_width

                            f.write(f"{str(data)} ")

        image = cv2.resize(image, (adjusted_width, adjusted_height))
        print(np.unique(image))
        return image, segmented_image_resize, mask, img_with_bbox, image_info['file_name']



    def process_info(self):
        self.info = self.coco['info']

    def process_images(self):
        self.images = {}
        for image in self.coco['images']:
            image_id = image['id']
            if image_id in self.images:
                print(f"ERROR: Skipping duplicate image id: {image}")
            else:
                self.images[image_id] = image

    def process_segmentations(self):
        self.segmentations = {}
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

