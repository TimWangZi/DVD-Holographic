import json
import unittest
from pathlib import Path
import numpy as np
from cdholo.calibration_target import normalize_origin,make_target,feature_mask
from cdholo.dvd_geometry import DvdSpiral

ROOT=Path(__file__).resolve().parents[1]


class CalibrationTargetTests(unittest.TestCase):
    def test_offset_gauge_preserves_physical_coordinates(self):
        cfg=json.loads((ROOT/'configs/tetrahedron_dvd.json').read_text())
        cfg['disc']['channel_offset_bits']=1234567
        cfg['disc']['start_angle_deg']=28.5
        normalized=normalize_origin(cfg)
        before=DvdSpiral(cfg['disc']);after=DvdSpiral(normalized['disc'])
        lbas=np.array([0,20000,270048],float)
        np.testing.assert_allclose(before.coordinates(lbas,733),after.coordinates(lbas,733),atol=2e-8)
        self.assertEqual(normalized['disc']['channel_offset_bits'],0.)

    def test_known_clock_outer_edges_match_generated_pattern(self):
        cfg=json.loads((ROOT/'configs/tetrahedron_dvd.json').read_text())
        pattern=json.loads((ROOT/'configs/calibration_dvd.json').read_text())
        target=make_target(cfg,pattern);g=DvdSpiral(target['nominal_config']['disc'])
        a=np.deg2rad(45)
        for feature in target['features']:
            if feature['kind']!='ring':continue
            rr=feature['nominal_radius_mm']
            self.assertAlmostEqual(float(np.hypot(*g.coordinates(0,feature['channel_index']))),rr,places=11)
            r=np.array([rr-.001,rr+.001])
            np.testing.assert_array_equal(feature_mask(r*np.cos(a),r*np.sin(a),pattern),[True,False])


if __name__=='__main__':unittest.main()
