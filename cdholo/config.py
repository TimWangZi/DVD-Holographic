import json
import numpy as np
from pathlib import Path
from .optics import tetrahedron


def load_config(path):
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    d,p,t,s=cfg['disc'],cfg['patch'],cfg['target'],cfg['simulation']
    medium=cfg.get('format','CD-R')
    if medium not in ('CD-R','DVD-R'): raise ValueError('Only CD-R and single-layer DVD-R are implemented')
    bit_mm=d['channel_bit_length_nm']*1e-6 if medium=='DVD-R' else d['linear_velocity_m_s']*1000/4321800
    if min(d['track_pitch_um'],bit_mm,d['program_start_radius_mm'],
           t['edge_mm'],t['distance_mm'],*p['size_mm'],*s['grid_step_um'])<=0:
        raise ValueError('Lengths, velocities and sampling steps must be positive')
    if d['spiral_direction'] not in (-1,1):
        raise ValueError('spiral_direction must be +1 or -1')
    if p['center_radius_mm']-p['size_mm'][1]/2<=d['program_start_radius_mm']+0.1:
        raise ValueError('Patch must be inside program area, with a 0.1 mm startup guard')
    if np.hypot(p['center_radius_mm']+p['size_mm'][1]/2,p['size_mm'][0]/2)>58:
        raise ValueError('Patch exceeds 58 mm outer radius')
    if t['points_per_edge']<2 or s['reconstruction_pixels']<16:
        raise ValueError('Need at least 2 points per edge and 16 reconstruction pixels')
    if cfg['media']['polarization'] not in ('s','p'):
        raise ValueError('polarization must be s or p')
    if cfg['optics']['beam_profile'] not in ('uniform','gaussian') or min(cfg['optics']['beam_radius_mm'])<=0:
        raise ValueError('Invalid beam profile or radius')
    if np.sum(np.sin(np.deg2rad(cfg['optics']['illumination_angles_deg']))**2)>1:
        raise ValueError('Illumination direction cosines do not define a physical direction')
    if not 0<cfg['optimization']['on_fraction']<1:
        raise ValueError('on_fraction must be between 0 and 1')
    values=[d['background_byte'],*cfg['optimization']['palette_bytes']]
    if any(not isinstance(v,int) or not 0<=v<=255 for v in values):
        raise ValueError('Data byte values must be integers in 0..255')
    if cfg['media']['edge_trim_um']<0 or not 0<cfg['media']['mark_width_um']<=d['track_pitch_um']:
        raise ValueError('Invalid mark geometry')
    verts,_,_=tetrahedron(t)
    if verts[:,2].min()<=0:
        raise ValueError('All target points must be in front of disc in this real-image mode')
    # Conservative spatial-frequency bound for the designed spherical wave.
    ext=np.max(np.abs(verts[:,:2]),axis=0)+np.array(p['size_mm'])/2
    wl=cfg['optics']['wavelength_nm']*1e-6
    freq=(ext/verts[:,2].min()+np.abs(np.sin(np.deg2rad(cfg['optics']['illumination_angles_deg']))))/wl
    effective=np.maximum(np.array(s['grid_step_um'])*1e-3,
                         [(16 if medium=='DVD-R' else 17)*bit_mm,d['track_pitch_um']*1e-3])
    if np.any(freq*effective>=0.5):
        raise ValueError(f'Target aliases at configured sampling: cycles/sample {freq*effective}; reduce angle/size or increase distance')
    out_step=s['reconstruction_span_mm']/(s['reconstruction_pixels']-1)
    if out_step>wl*(t['distance_mm']+min(s['focus_offsets_mm']))/(2*max(p['size_mm'])):
        raise ValueError('Reconstruction grid undersamples intensity; increase reconstruction_pixels')
    return cfg
