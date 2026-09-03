# ============================================================
# Multi-Face Tracker
# Simple IoU-based tracking
# ============================================================

import math


class MultiFaceTracker:

    def __init__(
        self,
        max_disappeared=15,
        max_distance=100
    ):

        self.next_id = 0

        self.objects = {}

        self.disappeared = {}

        self.max_disappeared = max_disappeared

        self.max_distance = max_distance


    # ========================================================
    # REGISTER
    # ========================================================

    def register(self, bbox):

        self.objects[self.next_id] = bbox

        self.disappeared[self.next_id] = 0

        self.next_id += 1


    # ========================================================
    # DEREGISTER
    # ========================================================

    def deregister(self, object_id):

        if object_id in self.objects:

            del self.objects[object_id]

        if object_id in self.disappeared:

            del self.disappeared[object_id]


    # ========================================================
    # CENTROID
    # ========================================================

    @staticmethod
    def centroid(bbox):

        x1, y1, x2, y2 = bbox

        cx = int((x1 + x2) / 2)

        cy = int((y1 + y2) / 2)

        return cx, cy


    # ========================================================
    # DISTANCE
    # ========================================================

    @staticmethod
    def distance(point1, point2):

        return math.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2
        )


    # ========================================================
    # UPDATE
    # ========================================================

    def update(self, detections):

        """
        detections:
            list of bounding boxes

            [
                (x1, y1, x2, y2),
                ...
            ]

        Returns:

            {
                track_id: bbox
            }
        """

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if len(detections) == 0:

            ids_to_delete = []

            for object_id in self.disappeared:

                self.disappeared[object_id] += 1

                if (
                    self.disappeared[object_id]
                    > self.max_disappeared
                ):

                    ids_to_delete.append(object_id)


            for object_id in ids_to_delete:

                self.deregister(object_id)


            return self.objects.copy()


        # ----------------------------------------------------
        # No existing objects
        # ----------------------------------------------------

        if len(self.objects) == 0:

            for bbox in detections:

                self.register(bbox)

            return self.objects.copy()


        # ----------------------------------------------------
        # Existing objects
        # ----------------------------------------------------

        object_ids = list(self.objects.keys())

        object_centroids = [
            self.centroid(self.objects[obj_id])
            for obj_id in object_ids
        ]

        detection_centroids = [
            self.centroid(bbox)
            for bbox in detections
        ]


        # ----------------------------------------------------
        # Calculate distances
        # ----------------------------------------------------

        distances = []

        for object_centroid in object_centroids:

            row = []

            for detection_centroid in detection_centroids:

                row.append(
                    self.distance(
                        object_centroid,
                        detection_centroid
                    )
                )

            distances.append(row)


        # ----------------------------------------------------
        # Match closest detections
        # ----------------------------------------------------

        used_objects = set()

        used_detections = set()

        matches = []


        while True:

            best_distance = float("inf")

            best_object = None

            best_detection = None


            for object_index in range(
                len(object_ids)
            ):

                if object_index in used_objects:

                    continue


                for detection_index in range(
                    len(detections)
                ):

                    if detection_index in used_detections:

                        continue


                    distance = distances[
                        object_index
                    ][
                        detection_index
                    ]


                    if distance < best_distance:

                        best_distance = distance

                        best_object = object_index

                        best_detection = detection_index


            if best_object is None:

                break


            if best_distance > self.max_distance:

                break


            matches.append(
                (
                    best_object,
                    best_detection
                )
            )

            used_objects.add(best_object)

            used_detections.add(best_detection)


        # ----------------------------------------------------
        # Update matched objects
        # ----------------------------------------------------

        for object_index, detection_index in matches:

            object_id = object_ids[object_index]

            self.objects[object_id] = detections[
                detection_index
            ]

            self.disappeared[object_id] = 0


        # ----------------------------------------------------
        # Handle disappeared objects
        # ----------------------------------------------------

        for object_index, object_id in enumerate(
            object_ids
        ):

            if object_index not in used_objects:

                self.disappeared[object_id] += 1

                if (
                    self.disappeared[object_id]
                    > self.max_disappeared
                ):

                    self.deregister(object_id)


        # ----------------------------------------------------
        # Register new detections
        # ----------------------------------------------------

        for detection_index, bbox in enumerate(
            detections
        ):

            if detection_index not in used_detections:

                self.register(bbox)


        return self.objects.copy()