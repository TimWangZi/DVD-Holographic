import json
import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np
from cdholo.config import load_config
from cdholo.geometry import Spiral
from cdholo.optics import tetrahedron, film_reflection, fresnel_roi
from cdholo.pipeline import export_wave, generate_pcm, selected_circ
from cdholo.encoding import DATA_POSITIONS

ROOT=Path(__file__).resolve().parents[1]


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(ROOT/'configs/tetrahedron.json')

    def test_regular_tetrahedron_after_rotation(self):
        vertices,points,_=tetrahedron(self.cfg['target'])
        lengths=[np.linalg.norm(vertices[i]-vertices[j]) for i in range(4) for j in range(i+1,4)]
        np.testing.assert_allclose(lengths,self.cfg['target']['edge_mm'],atol=1e-12)
        self.assertEqual(len(points),6*self.cfg['target']['points_per_edge'])

    def test_geometry_radius_inverse_and_track_pitch(self):
        s=Spiral(self.cfg['disc'])
        r=np.array([26.,28.,30.]); f=s.frame_at_radius(r)
        x,y=s.coordinates(f)
        np.testing.assert_allclose(np.hypot(x,y),r,atol=1e-12)
        x2,y2=s.coordinates(s.frame_at_radius(r+s.pitch))
        np.testing.assert_allclose(np.arctan2(y2,x2),np.arctan2(y,x),atol=2e-10)

    def test_thin_film_limits_and_passivity(self):
        self.assertAlmostEqual(film_reflection(1,[],1.5,0.0006328,0,'s'),-0.2)
        zero=[{'n_k':[2.11,.077],'thickness_nm':0}]
        self.assertAlmostEqual(film_reflection(1,zero,1.5,0.0006328,0,'s'),-0.2)
        lossy=[{'n_k':[2.11,.077],'thickness_nm':100}]
        self.assertLess(abs(film_reflection(1.58,lossy,.1838+3.4313j,.0006328,0,'s')),1)

    def test_fresnel_focus_location_and_sign(self):
        # An independently constructed converging quadratic wave must focus at +x,+y.
        x=np.linspace(-1,1,201); y=x.copy(); wl=.0006328; z=100.; px=.3; py=-.2
        xx,yy=np.meshgrid(x,y)
        wave=np.exp(-1j*np.pi/(wl*z)*((xx-px)**2+(yy-py)**2))
        xo=np.linspace(-.6,.6,121); yo=xo.copy()
        intensity=abs(fresnel_roi(wave,x,y,wl*1e6,z,xo,yo))**2
        row,col=np.unravel_index(intensity.argmax(),intensity.shape)
        self.assertAlmostEqual(xo[col],px,places=2); self.assertAlmostEqual(yo[row],py,places=2)

    def test_pcm_wave_endian_and_sector_alignment(self):
        canonical=np.tile(np.arange(24,dtype=np.uint8),(98,1))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.wav'; export_wave(path,canonical)
            with wave.open(str(path)) as f:
                self.assertEqual((f.getnchannels(),f.getsampwidth(),f.getframerate()),(2,2,44100))
                self.assertEqual(f.getnframes(),588)
                raw=f.readframes(f.getnframes())
            decoded=np.frombuffer(raw,np.uint8).reshape(-1,2)[:,::-1].copy().reshape(-1,24)
            np.testing.assert_array_equal(decoded,canonical)

    def test_spatial_precompensation_survives_circ(self):
        cfg=json.loads(json.dumps(self.cfg)); cfg['patch']['center_radius_mm']=25.25
        cfg['patch']['size_mm']=[.2,.2]; cfg['disc']['tail_seconds']=.05
        x=np.linspace(-.1,.1,51); y=x.copy(); xx,yy=np.meshgrid(x,y)
        mask=(xx*yy)>0
        pcm,selected,desired,inside,_=generate_pcm(cfg,x,y,mask)
        f2=selected_circ(pcm,selected,4096)
        np.testing.assert_array_equal(f2[:,DATA_POSITIONS][inside],desired[inside])


if __name__=='__main__': unittest.main()
