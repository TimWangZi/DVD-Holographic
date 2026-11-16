import copy
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cdholo.config import load_config
from cdholo.dvd_geometry import DvdSpiral,BITS_PER_SECTOR,payload_channel_centers,recording_channel_centers
from cdholo.dvd_encoding import PAYLOAD_RECORDING_OFFSETS
from cdholo.dvd_pipeline import create_payload,forward_aperture
from cdholo.iso9660 import create_image

ROOT=Path(__file__).resolve().parents[1]


class DvdPipelineTests(unittest.TestCase):
    def test_iso9660_volume_and_fixed_extent(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'image.iso';create_image(path,128)
            data=path.read_bytes(); pvd=data[16*2048:17*2048]
            self.assertEqual(pvd[:7],b'\x01CD001\x01')
            self.assertEqual(struct.unpack('<I',pvd[80:84])[0],128)
            self.assertEqual(struct.unpack('>I',pvd[84:88])[0],128)
            self.assertEqual(struct.unpack('<H',pvd[128:130])[0],2048)
            self.assertEqual(data[17*2048:17*2048+7],b'\xffCD001\x01')
            root=data[20*2048:21*2048]; pos=0; entries={}
            while root[pos]:
                rec=root[pos:pos+root[pos]]; name=rec[33:33+rec[32]]
                entries[name]=(struct.unpack('<I',rec[2:6])[0],struct.unpack('<I',rec[10:14])[0])
                pos+=len(rec)
            self.assertEqual(entries[b'HOLOGRAM.BIN;1'],(32,96*2048))
            self.assertIn(b'README.TXT;1',entries)
            self.assertEqual(list(entries),sorted(entries))

    def test_spiral_and_payload_clock_correspondence(self):
        cfg=load_config(ROOT/'configs/tetrahedron_dvd.json'); g=DvdSpiral(cfg['disc'])
        r=np.array([24.5,28.,30.]); lbas=g.sector_at_radius(r)
        np.testing.assert_allclose(np.hypot(*g.coordinates(lbas)),r,atol=1e-12)
        # Windows defaults scalar NumPy integers to int32 in some NumPy versions.
        # A DVD's channel clock index exceeds 2**31 well inside the data zone.
        n=270048
        expected=np.sqrt(g.r0**2+g.pitch*(n*38688)*g.bit_mm/np.pi)
        self.assertAlmostEqual(float(np.hypot(*g.coordinates(n))),expected,places=11)
        np.testing.assert_allclose(g.coordinates(n),g.coordinates(np.array(n,dtype=np.int64)),atol=1e-12)
        np.testing.assert_allclose(payload_channel_centers(),recording_channel_centers()[PAYLOAD_RECORDING_OFFSETS])
        self.assertEqual(BITS_PER_SECTOR,26*(32+91*16))

    def test_iso_to_encoded_spatial_target(self):
        cfg=copy.deepcopy(load_config(ROOT/'configs/tetrahedron_dvd.json'))
        cfg['patch']['center_radius_mm']=24.25; cfg['patch']['size_mm']=[.1,.05]
        x=np.linspace(-.05,.05,51); y=np.linspace(-.025,.025,51)
        xx,yy=np.meshgrid(x,y); mask=(xx*yy)>0
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.iso'
            active,meta=create_payload(cfg,path,x,y,mask)
            marks=forward_aperture(cfg,path,x,y,mask,active,meta)
            self.assertGreater(meta['controlled_payload_bytes'],100)
            self.assertEqual(meta['scrambler_mismatches'],0)
            self.assertEqual(meta['encoded_payload_mismatches'],0)
            self.assertEqual(meta['encoded_payload_bytes_checked'],meta['controlled_payload_bytes'])
            self.assertTrue(np.isfinite(marks).all())
            self.assertGreater(float(np.ptp(marks)),0)
            with path.open('rb') as f:
                f.seek(16*2048);self.assertEqual(f.read(7),b'\x01CD001\x01')


if __name__=='__main__':unittest.main()
