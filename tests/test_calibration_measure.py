import csv
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image

from cdholo.calibration_measure import make_transform, transform_points, observation_rows, measure_photo


class CalibrationMeasureTests(unittest.TestCase):
    def setUp(self):
        self.camera = np.array([[8., 1.2], [.5, -8.]])
        self.origin = np.array([600., 600.])
        self.angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
        self.circle = np.column_stack((np.cos(self.angles), np.sin(self.angles)))
        self.rim = self.to_pixel(self.circle*60)
        self.hole = self.to_pixel(self.circle*7.5)

    def to_pixel(self, points):
        return np.asarray(points) @ self.camera.T + self.origin

    def test_affine_normalization_recovers_metric_up_to_rotation(self):
        transform, qa, _, _ = make_transform(self.rim, self.hole, 120)
        probes = np.array([[0., 0.], [20., 10.], [-15., 31.], [7., -4.]])
        recovered = transform_points(self.to_pixel(probes), transform)
        np.testing.assert_allclose(np.linalg.norm(recovered, axis=1), np.linalg.norm(probes, axis=1), atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(recovered[1:]-recovered[:-1], axis=1),
                                   np.linalg.norm(probes[1:]-probes[:-1], axis=1), atol=1e-8)
        self.assertLess(qa['rim_fit_rms_px'], 1e-8)
        self.assertAlmostEqual(qa['measured_hole_diameter_mm'], 15, places=7)
        self.assertFalse(qa['perspective_rectification'])
        self.assertFalse(qa['image_mirroring_verified'])

    def test_hole_is_independent_check_and_trace_order_is_enforced(self):
        transform, qa, _, _ = make_transform(self.rim, self.hole+[8., 0.], 120)
        self.assertGreater(qa['rim_hole_centre_offset_mm'], .9)
        self.assertTrue(any('centres disagree' in warning for warning in qa['warning_messages']))
        transform, qa, _, _ = make_transform(self.rim, self.hole, 120)
        target = {'features': [{'id': 'S01', 'kind': 'spoke', 'nominal_angle_deg': 0.}]}
        radii = np.linspace(25, 33, 12)
        points = self.to_pixel(np.column_stack((radii, np.zeros_like(radii))))
        rows, _, _ = observation_rows(target, {'S01': points.tolist()}, transform, .1)
        self.assertEqual(len(rows), 12)
        with self.assertRaisesRegex(ValueError, 'radius must increase'):
            observation_rows(target, {'S01': points[::-1].tolist()}, transform, .1)

    def test_saved_click_replay_produces_csv_qa_and_preview(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root/'photo.png'; Image.new('RGB', (1200, 1200), '#252525').save(image)
            target = {'schema': 'dvd-calibration-v1', 'features': [
                {'id': 'R01', 'kind': 'ring', 'nominal_radius_mm': 25.08, 'channel_index': 1e8},
                {'id': 'S01', 'kind': 'spoke', 'nominal_angle_deg': 0.}]}
            radii = np.linspace(25.2, 32.4, 16); theta = np.deg2rad(np.linspace(20, 150, 16))
            session = {'schema': 'dvd-calibration-clicks-v1', 'outer_rim_px': self.rim.tolist(),
                'inner_hole_px': self.hole.tolist(), 'features': {
                    'R01': self.to_pixel(self.circle*25.08).tolist(),
                    'S01': self.to_pixel(np.column_stack((radii*np.cos(theta), radii*np.sin(theta)))).tolist()}}
            (root/'target.json').write_text(json.dumps(target), encoding='utf-8')
            (root/'clicks.json').write_text(json.dumps(session), encoding='utf-8')
            output = root/'measurements.csv'
            qa = measure_photo(image, root/'target.json', output, clicks_path=root/'clicks.json')
            self.assertEqual(qa['observations_count'], 40)
            self.assertAlmostEqual(qa['features']['R01']['median_radius_mm'], 25.08, places=7)
            with output.open(encoding='utf-8', newline='') as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 40)
            self.assertEqual(rows[0]['angle_unwrapped_deg'], '')
            self.assertNotEqual(rows[-1]['angle_unwrapped_deg'], '')
            for suffix in ('.clicks.json', '.qa.json', '.preview.png'):
                self.assertTrue(output.with_suffix(suffix).exists())


if __name__ == '__main__': unittest.main()
