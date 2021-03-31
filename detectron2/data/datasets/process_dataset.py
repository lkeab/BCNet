# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np
import cv2
from PIL import Image
from skimage import measure

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes, polygons_to_bitmask
from fvcore.common.file_io import PathManager, file_lock


from .. import MetadataCatalog, DatasetCatalog

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_coco_json", "load_sem_seg"]

CLASS_NAMES = [
    "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
    "potted plant", "sheep", "bench", "train", "tv",
]

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        print('meta:', meta)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    sum_box = 0
    sum_co_box = 0
    intersect_rate = 0.0
    intersect_num = 0

    num_instances_without_valid_segmentation = 0
    index_c = 0

    for jdex, (img_dict, anno_dict_list) in enumerate(imgs_anns):
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        print('file name:', jdex, ':', record["file_name"])
        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        seg_list = []
        for obj in objs:
            seg_list.append(obj['segmentation'])
        #print('seg list:', seg_list)

        dirname = "mask-vis"
        os.makedirs(dirname, exist_ok=True)

        bitmask_list = []
        if len(seg_list) > 0:
            for index, seg in enumerate(seg_list):
                #print('seg len:', len(seg))
                invalid = False
                for sub_seg in seg:
                    #print('seg len:', len(sub_seg))
                    if len(sub_seg) < 6:
                        invalid = True
                if not invalid:
                    bitmask = polygons_to_bitmask(seg, img_dict["height"], img_dict["width"])
                else:
                    bitmask = np.zeros((int(img_dict["height"]), int(img_dict["width"])), dtype=bool)

                bitmask_list.append(bitmask.astype('int'))


        box_list = []
        for obj in objs:
            box_list.append([obj['bbox'][0],obj['bbox'][1],obj['bbox'][0]+obj['bbox'][2],obj['bbox'][1]+obj['bbox'][3]])

        box_mask_list = []
        for index, obj in enumerate(objs):
            box_mask = np.zeros((int(img_dict["height"]), int(img_dict["width"])), dtype=int)
            box_mask[int(box_list[index][1]):int(box_list[index][3]),int(box_list[index][0]):int(box_list[index][2])] = 1
            box_mask_list.append(box_mask)


        sum_box += len(box_list)

        for index1, a_box in enumerate(box_list):
            union_mask_whole = np.zeros((int(img_dict["height"]), int(img_dict["width"])), dtype=int)
            for index2, b_box in enumerate(box_list):
                if index1 != index2:
                    iou = bb_intersection_over_union(a_box, b_box)
                    if iou > 0.05:
                        union_mask = np.multiply(box_mask_list[index1], bitmask_list[index2])
                        union_mask_whole += union_mask
            
            print("===========================================")
            print('bit mask area:', bitmask_list[index1].sum())
            union_mask_whole[union_mask_whole > 1.0] = 1.0
            print('cropped union mask area:', union_mask_whole.sum())
            intersect_mask = union_mask_whole * bitmask_list[index1]
            print('intersect mask area:', intersect_mask.sum()) 
            print('intersect rate:', intersect_mask.sum()/float(bitmask_list[index1].sum()))
            print("===========================================")
            
            if intersect_mask.sum() >= 1.0:
                intersect_num += 1

            if float(bitmask_list[index1].sum()) > 1.0:
                intersect_rate += intersect_mask.sum()/float(bitmask_list[index1].sum())
            
            union_mask_non_zero_num = np.count_nonzero(union_mask_whole.astype(int))
            record["annotations"][index1]['bg_object_segmentation'] = []
            if union_mask_non_zero_num > 20:
                sum_co_box += 1
                contours = measure.find_contours(union_mask_whole.astype(int), 0)
                for contour in contours:
                    if contour.shape[0] > 500: 
                        contour = np.flip(contour, axis=1)[::10,:]
                    elif contour.shape[0] > 200: 
                        contour = np.flip(contour, axis=1)[::5,:]
                    elif contour.shape[0] > 100: 
                        contour = np.flip(contour, axis=1)[::3,:]
                    elif contour.shape[0] > 50: 
                        contour = np.flip(contour, axis=1)[::2,:]
                    else:
                        contour = np.flip(contour, axis=1)

                    segmentation = contour.ravel().tolist()
                    record["annotations"][index1]['bg_object_segmentation'].append(segmentation)

        dataset_dicts.append(record)
        if jdex > 10000:
            break

    print('sum intersect rate:', intersect_rate)
    print('sum box:', sum_box)

    avg_intersect_rate = intersect_rate/float(sum_box)
    avg_intersect_rate_over_inter = intersect_rate/float(intersect_num)
    print('avg rate:', avg_intersect_rate)
    print('avg rate over intersect:', avg_intersect_rate_over_inter)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def load_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = sorted(
        (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
        key=lambda file_path: file2id(image_root, file_path),
    )
    gt_files = sorted(
        (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
        key=lambda file_path: file2id(gt_root, file_path),
    )

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)

    # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    if len(input_files) != len(gt_files):
        logger.warn(
            "Directory {} and {} has {} and {} files, respectively.".format(
                image_root, gt_root, len(input_files), len(gt_files)
            )
        )
        input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
        gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        # sort, otherwise each worker may obtain a list[dict] in different order
        intersect = sorted(intersect)
        logger.warn("Will use their intersection of {} files.".format(len(intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]

    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    dataset_dicts = []
    for (img_path, gt_path) in zip(input_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def convert_to_coco_dict(dataset_dicts, dataset_name):
    """
    Convert a dataset in detectron2's standard format into COCO json format

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name:
            name of the source dataset
            must be registered in DatastCatalog and in detectron2's standard format
    Returns
        coco_dict: serializable dict in COCO json format
    """

    #dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        print('metadata.thing_dataset_id_to_contiguous_id:', metadata.thing_dataset_id_to_contiguous_id)
        print('reverse_id_mapping:', reverse_id_mapping)
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
        print('continue')
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    print('categories:', categories)
    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format
            bbox = annotation["bbox"]
            bbox_mode = annotation["bbox_mode"]
            bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)
            #print('annotation:', annotation)
            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                #if len(segmentation) >= 1:
                #    polygons = PolygonMasks([segmentation])
                #    area = polygons.area()[0].item()
                #else:
                bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()
            else:
                # Computing areas using bounding boxes
                bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                area = Boxes([bbox_xy]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = area
            coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)
            coco_annotation["category_id"] = reverse_id_mapper(annotation["category_id"])


            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                coco_annotation["segmentation"] = annotation["segmentation"]
            if "bg_object_segmentation" in annotation:
                coco_annotation["bg_object_segmentation"] = annotation["bg_object_segmentation"]


            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"num images: {len(coco_images)}, num annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.info(f"Cached annotations in COCO format already exist: {output_file}")
        else:
            logger.info(f"Converting dataset annotations in '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name)

            with PathManager.open(output_file, "w") as json_file:
                logger.info(f"Caching annotations in COCO format: {output_file}")
                json.dump(coco_dict, json_file)


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))
    coco_dict = convert_to_coco_dict(dicts, sys.argv[3])
    #output_file = 'instances_train_2017_transform_slight_map.json'
    #output_file = 'test.json'
    #with open(output_file, "w") as json_file:
    #    logger.info(f"Caching annotations in COCO format: {output_file}")
    #    json.dump(coco_dict, json_file)


    '''
    dirname = "coco-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
    '''
