import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cdholo.calibration_fit import fit_calibration

ROOT=Path(__file__).resolve().parents[1]


def synthetic_files(directory,*,spoke_count=24,ring_count=3,noise=0.,seed=12,actual_direction=1,spoke_angles=(0.,120.)):
    cfg=json.loads((ROOT/'configs/tetrahedron_dvd.json').read_text())
    nom=cfg['disc']; r0n=nom['program_start_radius_mm']; pn=nom['track_pitch_um']*1e-3
    bn=nom['channel_bit_length_nm']*1e-6; Bn=pn*bn/np.pi
    truth={'r0':23.998,'pitch_um':.74002,'bit_nm':133.301,'angle_deg':27.,'cx':.015,'cy':-.012}
    p=truth['pitch_um']*1e-3;b=truth['bit_nm']*1e-6;B=p*b/np.pi
    features=[];rows=[];rng=np.random.default_rng(seed)
    for i,rn in enumerate(np.linspace(25.5,31.,ring_count)):
        fid=f'R{i+1:02d}';N=(rn*rn-r0n*r0n)/Bn
        features.append({'id':fid,'kind':'ring','nominal_radius_mm':rn,'channel_index':N})
        R=np.sqrt(truth['r0']**2+B*N)
        for j,a in enumerate(np.arange(12)*30):
            xx=R*np.cos(np.deg2rad(a))+truth['cx']+rng.normal(0,noise)
            yy=R*np.sin(np.deg2rad(a))+truth['cy']+rng.normal(0,noise)
            rows.append({'feature_id':fid,'kind':'ring','click_index':j,'x_mm':xx,'y_mm':yy,
                         'radius_mm':np.hypot(xx,yy),'theta_deg':np.rad2deg(np.arctan2(yy,xx)),
                         'angle_unwrapped_deg':'','sigma_mm':max(noise,.001)})
    for i,phi in enumerate(spoke_angles):
        fid=f'S{i+1:02d}';features.append({'id':fid,'kind':'spoke','nominal_angle_deg':phi})
        for j,rn in enumerate(np.linspace(25.5,31.,spoke_count)):
            N=(rn*rn-r0n*r0n)/Bn;R=np.sqrt(truth['r0']**2+B*N)
            theta=truth['angle_deg']+actual_direction*phi+actual_direction*360*((R-truth['r0'])/p-(rn-r0n)/pn)
            xx=R*np.cos(np.deg2rad(theta))+truth['cx']+rng.normal(0,noise)
            yy=R*np.sin(np.deg2rad(theta))+truth['cy']+rng.normal(0,noise)
            raw=np.rad2deg(np.arctan2(yy,xx))
            unwrap=raw+360*round((theta-raw)/360)
            rows.append({'feature_id':fid,'kind':'spoke','click_index':j,'x_mm':xx,'y_mm':yy,
                         'radius_mm':np.hypot(xx,yy),'theta_deg':raw,
                         'angle_unwrapped_deg':unwrap,'sigma_mm':max(noise,.001)})
    target={'schema':'dvd-calibration-v1','nominal_config':cfg,'features':features,
            'measurement':{'disc_diameter_mm':120.,'sigma_mm':.001}}
    target_path=Path(directory)/'target.json';target_path.write_text(json.dumps(target),encoding='utf-8')
    csv_path=Path(directory)/'observations.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return target_path,csv_path,truth


class CalibrationFitTests(unittest.TestCase):
    def test_joint_geometry_and_centre_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            target,observations,truth=synthetic_files(d)
            result=fit_calibration(target,observations,Path(d)/'fit')
            fitted=result['parameters']
            self.assertAlmostEqual(fitted['effective_program_start_radius_mm'],truth['r0'],places=6)
            self.assertAlmostEqual(fitted['track_pitch_um'],truth['pitch_um'],places=6)
            self.assertAlmostEqual(fitted['channel_bit_length_nm'],truth['bit_nm'],places=4)
            np.testing.assert_allclose(fitted['photo_center_correction_mm'],[truth['cx'],truth['cy']],atol=1e-7)
            self.assertLess(result['held_out_validation']['tangential_rms_um'],.001)
            exported=json.loads((Path(d)/'fit/calibrated_config.json').read_text())
            self.assertEqual(exported['disc']['channel_offset_bits'],0.)
            self.assertEqual(exported['disc']['start_angle_deg'],0.)
            self.assertFalse(exported['calibration']['material_parameters_measured'])
            self.assertFalse(result['winding_ambiguity_excluded'])

    def test_coarse_fixes_bit_length_and_identifies_product(self):
        with tempfile.TemporaryDirectory() as d:
            target,observations,truth=synthetic_files(d,spoke_count=0)
            result=fit_calibration(target,observations,Path(d)/'fit',mode='coarse')
            p=result['parameters']
            self.assertEqual(p['channel_bit_length_nm'],133.3)
            self.assertAlmostEqual(p['track_pitch_um']*p['channel_bit_length_nm'],truth['pitch_um']*truth['bit_nm'],places=6)
            self.assertTrue(result['conditional_standard_errors']['bit_length_fixed'])
            self.assertIsNone(result['held_out_validation'])

    def test_insufficient_anchors_and_trace_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            target,observations,_=synthetic_files(d,ring_count=1)
            with self.assertRaisesRegex(ValueError,'two separated'):
                fit_calibration(target,observations,Path(d)/'fit')
            target,observations,_=synthetic_files(d,spoke_count=7)
            with self.assertRaisesRegex(ValueError,'eight ordered'):
                fit_calibration(target,observations,Path(d)/'fit')

    def test_measurement_noise_is_reported_not_zero_precision(self):
        with tempfile.TemporaryDirectory() as d:
            target,observations,truth=synthetic_files(d,noise=.002,spoke_count=32)
            result=fit_calibration(target,observations,Path(d)/'fit')
            self.assertGreater(result['residuals']['ring_radial_rms_um'],.1)
            self.assertGreater(result['conditional_standard_errors']['effective_radius_mm'],0.)
            self.assertGreater(result['held_out_validation']['tangential_rms_um'],0.)
            self.assertFalse(result['hologram_registration_reference']['within_heuristic_budget'])
            self.assertLess(abs(result['parameters']['channel_bit_length_nm']-truth['bit_nm']),.02)

    def test_mirrored_spiral_is_inferred_and_exported(self):
        with tempfile.TemporaryDirectory() as d:
            target,observations,truth=synthetic_files(d,actual_direction=-1)
            result=fit_calibration(target,observations,Path(d)/'fit')
            p=result['parameters']
            self.assertEqual(p['spiral_direction'],-1)
            self.assertAlmostEqual(p['track_pitch_um'],truth['pitch_um'],places=6)
            self.assertAlmostEqual(p['channel_bit_length_nm'],truth['bit_nm'],places=4)
            self.assertLess(result['direction_inference']['circular_phase_rms_deg'],1e-6)
            exported=json.loads((Path(d)/'fit/calibrated_config.json').read_text())
            self.assertEqual(exported['disc']['spiral_direction'],-1)
            self.assertEqual(json.loads(target.read_text())['nominal_config']['disc']['spiral_direction'],1)

    def test_one_spoke_requires_explicit_direction(self):
        with tempfile.TemporaryDirectory() as d:
            target,observations,truth=synthetic_files(d,actual_direction=-1)
            with observations.open(newline='',encoding='utf-8') as f:
                reader=csv.DictReader(f);fields=reader.fieldnames;rows=[r for r in reader if r['feature_id']!='S02']
            with observations.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            with self.assertRaisesRegex(ValueError,'two asymmetric'):
                fit_calibration(target,observations,Path(d)/'fit')
            result=fit_calibration(target,observations,Path(d)/'fit',direction='-1')
            self.assertEqual(result['parameters']['spiral_direction'],-1)
            self.assertAlmostEqual(result['parameters']['track_pitch_um'],truth['pitch_um'],places=6)

    def test_diametric_spokes_do_not_claim_chirality(self):
        with tempfile.TemporaryDirectory() as d:
            target,observations,_=synthetic_files(d,spoke_angles=(0.,180.))
            with self.assertRaisesRegex(ValueError,'uniquely identify direction'):
                fit_calibration(target,observations,Path(d)/'fit')


if __name__=='__main__':unittest.main()
