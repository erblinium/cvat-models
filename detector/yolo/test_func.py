import unittest
from types import SimpleNamespace

from func import (
    DEFAULT_KEYPOINT_THRESHOLD,
    YoloPoseEstimationFunction,
    _expand_legacy_keypoint_names,
)


class TensorValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return list(self.value)


class YoloPoseEstimationFunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.function = YoloPoseEstimationFunction.__new__(YoloPoseEstimationFunction)
        self.function._keypoint_names = [str(index) for index in range(1, 14)]
        self.function._keypoint_threshold = DEFAULT_KEYPOINT_THRESHOLD

    def test_expands_numeric_legacy_names(self) -> None:
        self.assertEqual(
            _expand_legacy_keypoint_names([str(index) for index in range(1, 13)]),
            [str(index) for index in range(1, 14)],
        )

    def test_uses_person_13_skeleton_and_person_box_labels(self) -> None:
        skeleton, box = self.function._label_specs("person", 4)

        self.assertEqual(skeleton.name, "person-13")
        self.assertEqual([label.name for label in skeleton.sublabels], self.function._keypoint_names)
        self.assertEqual(box.name, "person-box")

    def test_adds_visibility_one_head_for_legacy_model(self) -> None:
        box = TensorValue([10.0, 20.0, 30.0, 80.0])
        keypoints = [TensorValue([float(index), float(index + 1)]) for index in range(12)]
        confidences = [TensorValue(0.9) for _ in range(12)]

        annotations = list(
            self.function._keypoint_annotations(box, keypoints, confidences)
        )

        self.assertEqual(len(annotations), 13)
        head = annotations[0]
        self.assertEqual(head.label_id, 0)
        self.assertEqual(head.points, [20.0, 20.0])
        self.assertFalse(head.outside)
        self.assertTrue(head.occluded)
        self.assertEqual(annotations[1].label_id, 1)
        self.assertEqual(annotations[-1].label_id, 12)

    def test_native_13_keypoints_pass_through(self) -> None:
        box = TensorValue([10.0, 20.0, 30.0, 80.0])
        keypoints = [TensorValue([float(index), float(index + 1)]) for index in range(13)]
        confidences = [TensorValue(0.9) for _ in range(13)]

        annotations = list(
            self.function._keypoint_annotations(box, keypoints, confidences)
        )

        self.assertEqual(len(annotations), 13)
        self.assertEqual(annotations[0].label_id, 0)
        self.assertEqual(annotations[0].points, [0.0, 1.0])
        self.assertFalse(annotations[0].outside)

    def test_uses_configured_keypoint_threshold(self) -> None:
        self.function._keypoint_threshold = 0.75
        annotations = list(
            self.function._keypoint_annotations(
                TensorValue([10.0, 20.0, 30.0, 80.0]),
                [TensorValue([float(index), float(index + 1)]) for index in range(13)],
                [TensorValue(0.74), *[TensorValue(0.75) for _ in range(12)]],
            )
        )

        self.assertTrue(annotations[0].outside)
        self.assertFalse(annotations[1].outside)

    def test_disables_skeleton_at_keypoint_threshold_one_or_higher(self) -> None:
        results = SimpleNamespace(
            boxes=SimpleNamespace(
                cls=[TensorValue(0)],
                xyxy=[TensorValue([10.0, 20.0, 30.0, 80.0])],
            ),
            keypoints=SimpleNamespace(
                xy=[
                    [
                        TensorValue([float(index), float(index + 1)])
                        for index in range(13)
                    ]
                ],
                conf=[[TensorValue(0.99) for _ in range(13)]],
            ),
        )

        for threshold in (1.0, 1.5):
            with self.subTest(threshold=threshold):
                self.function._keypoint_threshold = threshold
                annotations = list(self.function._annotations_from_results(results))

                self.assertEqual(len(annotations), 1)
                self.assertEqual(str(annotations[0].type), "rectangle")

    def test_rejects_invalid_keypoint_threshold(self) -> None:
        for threshold in (-0.01, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(
                    ValueError, "keypoint_threshold must be a non-negative number"
                ):
                    YoloPoseEstimationFunction(None, keypoint_threshold=threshold)

    def test_rejects_other_model_schema_mismatches(self) -> None:
        with self.assertRaisesRegex(ValueError, "Model returned 11 keypoints"):
            list(
                self.function._keypoint_annotations(
                    TensorValue([10.0, 20.0, 30.0, 80.0]),
                    [TensorValue([0.0, 0.0]) for _ in range(11)],
                    [TensorValue(0.9) for _ in range(11)],
                )
            )


if __name__ == "__main__":
    unittest.main()
