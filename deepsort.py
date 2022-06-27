# comment out below line to enable tensorflow logging outputs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import cv2
import time
import numpy as np
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

iou_th = 0.45
score_th = 0.25
model = './checkpoints/yolov4-tiny-416'
# video = 0  # for camera
video = 'los.avi'  # video path


def main(_argv):
    # ## comment out for video writting ##
    # out = cv2.VideoWriter('deepsort.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280, 720))
    # ## Definition of the parameters ##
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    color = (138, 22, 128)
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
    input_size = 416
    video_path = video
    saved_model_loaded = tf.saved_model.load(model, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    vid = cv2.VideoCapture(video_path)
    frame_num = 0
    while True:
        ret, frame = vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended, try a different video format!')
            break
        frame_num += 1
        image_data = cv2.resize(frame, (input_size, input_size))  # 416X416
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou_th,
            score_threshold=score_th
        )
        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        allowed_classes = list(class_names.values())
        # allowed_classes = ['car']
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                                    score, class_name, feature in zip(bboxes, scores, names, features)]
        # run non-max supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        tracker.predict()
        tracker.update(detections)
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            num = track.track_id
            class_name = track.get_class()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]),
                                                                int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-16)),
                              (int(bbox[0])+(len(class_name)+len(str(num)))*10,
                               int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(num),
                            (int(bbox[0]), int(bbox[1]-5)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
        # calculate frames per second of running detections
        fps = int(1.0 / (time.time() - start_time))
        result = np.asarray(frame)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.rectangle(result, (0, 0), (110, 30), (250, 250, 0), -1)
        cv2.putText(result, 'FPS-' + str(fps), (5, 25),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (10, 16, 24), 2)
        cv2.imshow("Output Video", result)
        # out.write(result)
        if cv2.waitKey(1) == 13:
            break
    vid.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
