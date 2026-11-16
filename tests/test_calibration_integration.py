"""Target -> independent synthetic camera -> click replay -> geometry fitting."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from cdholo.calibration_target import make_target
from cdholo.calibration_measure import measure_photo
from cdholo.calibration_fit import fit_calibration

ROOT=Path(__file__).resolve().parents[1]


class CalibrationIntegrationTests(unittest.TestCase):
    def test_mirrored_spiral_recovered_through_photo_measurement(self):
        cfg=json.loads((ROOT/'configs/tetrahedron_dvd.json').read_text())
        pattern=json.loads((ROOT/'configs/calibration_dvd.json').read_text())
        target=make_target(cfg,pattern)
        nominal=cfg['disc'];rn0=nominal['program_start_radius_mm'];pn=nominal['track_pitch_um']*1e-3
        Bn=pn*nominal['channel_bit_length_nm']*1e-6/np.pi
        r0=23.998;p=.74002e-3;b=133.301e-6;B=p*b/np.pi
        camera=np.array([[15.,1.1],[.8,-15.]]);origin=np.array([1000.,1000.])
        def pixel(points):return (np.asarray(points)@camera.T+origin).tolist()
        a=np.arange(24)*2*np.pi/24;unit=np.column_stack([np.cos(a),np.sin(a)])
        session={'schema':'dvd-calibration-clicks-v1','outer_rim_px':pixel(unit*60),
                 'inner_hole_px':pixel(unit*7.5),'features':{}}
        for f in target['features']:
            if f['kind']=='ring':
                if f['id'] not in ('R01','R05','R09'):continue
                radius=np.sqrt(r0*r0+B*f['channel_index'])
                points=unit*radius
            else:
                source_r=np.linspace(25.2,32.8,48)
                clocks=(source_r*source_r-rn0*rn0)/Bn
                radius=np.sqrt(r0*r0+B*clocks)
                # Actual spiral clockwise, opposite the nominal target +1 direction.
                theta=np.deg2rad(27-f['nominal_angle_deg']-360*((radius-r0)/p-(source_r-rn0)/pn))
                points=np.column_stack([radius*np.cos(theta),radius*np.sin(theta)])
            session['features'][f['id']]=pixel(points+[.015,-.012])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);photo=root/'synthetic.png';Image.new('RGB',(2000,2000),'gray').save(photo)
            (root/'target.json').write_text(json.dumps(target),encoding='utf-8')
            (root/'clicks.json').write_text(json.dumps(session),encoding='utf-8')
            measure_photo(photo,root/'target.json',root/'observations.csv',clicks_path=root/'clicks.json')
            report=fit_calibration(root/'target.json',root/'observations.csv',root/'fit')
            updated=json.loads((root/'fit/calibrated_config.json').read_text())
            self.assertEqual(updated['disc']['spiral_direction'],-1)
            self.assertAlmostEqual(updated['disc']['program_start_radius_mm'],r0,places=5)
            self.assertAlmostEqual(updated['disc']['track_pitch_um'],p*1000,places=5)
            self.assertAlmostEqual(updated['disc']['channel_bit_length_nm'],b*1e6,places=3)
            self.assertFalse(updated['calibration']['material_parameters_measured'])


if __name__=='__main__':unittest.main()
