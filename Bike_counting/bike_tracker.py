import cv2
import os
import torch 
import threading
import numpy as np
from datetime import datetime
from shapely.geometry import Point, Polygon, LineString
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors

class ObjectCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_count = 0
        self.out_count = 0
        self.counted_ids = []
        self.classwise_counts = {}
        self.region_initialized = False
        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.margin = self.line_width * 2

        # === Daily folder logic ===
        today_str = datetime.now().strftime("%Y-%m-%d")
        self.output_dir = os.path.join("count_logs", today_str)
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "object_counts.txt")

        self.Polygon = Polygon
        self.Point = Point
        self.LineString = LineString

    def count_objects(self, current_centroid, track_id, prev_position, cls, im0, box):
        if prev_position is None or track_id in self.counted_ids:
            return

        crossed = False
        direction = None

        if len(self.region) == 2:
            line = self.LineString(self.region)
            if line.intersects(self.LineString([prev_position, current_centroid])):
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    direction = "IN" if current_centroid[0] > prev_position[0] else "OUT"
                else:
                    direction = "IN" if current_centroid[1] > prev_position[1] else "OUT"
                crossed = True

        elif len(self.region) > 2:
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)
                direction = "IN" if (
                    region_width < region_height and current_centroid[0] > prev_position[0]
                    or region_width >= region_height and current_centroid[1] > prev_position[1]
                ) else "OUT"
                crossed = True

        if crossed and direction:
            self.classwise_counts[self.names[cls]][direction] += 1
            self.counted_ids.append(track_id)

            # Save cropped image
            x1, y1, x2, y2 = map(int, box)
            crop = im0[y1:y2, x1:x2]
            timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            class_name = self.names[cls]
            image_name = f"{track_id}_{class_name}_{direction}_{timestamp_filename}.jpg"
            image_path = os.path.join(self.output_dir, image_name)
            cv2.imwrite(image_path, crop)

            # Log
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(f"{timestamp}, TrackID: {track_id}, Class: {class_name}, Direction: {direction}\n")

    def store_classwise_counts(self, cls):
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, plot_im):
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
                                 f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0):
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)
        self.annotator.draw_region(self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            self.store_tracking_history(track_id, box)
            self.store_classwise_counts(cls)

            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

            self.count_objects(current_centroid, track_id, prev_position, cls, im0, box)

        plot_im = self.annotator.result()
        self.display_counts(plot_im)
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=self.classwise_counts,
            total_tracks=len(self.track_ids),
        )

    def extract_tracks(self, im0):
        results = self.model.track(
            im0,
            persist=True,
            conf=self.CFG["conf"],
            iou=self.CFG["iou"],
            classes=[1, 3]  # bicycle and motorcycle only
        )

        self.boxes = []
        self.track_ids = []
        self.clss = []

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue

                 # add at the top if not present
                xyxy = box.xyxy[0].cpu()  # ✅ Keep it as a PyTorch tensor
  # ✅ NumPy array to avoid `.numel` error
                self.boxes.append(xyxy)
                self.track_ids.append(track_id)
                self.clss.append(cls_id)
