"""Known-address DVD-R rings and spokes for geometric calibration."""
import copy
import csv
import json
import time
from pathlib import Path
import numpy as np
from .dvd_geometry import DvdSpiral,payload_channel_centers,BITS_PER_SECTOR
from .dvd_encoding import scramble_payload,encode_ecc_blocks,PAYLOAD_RECORDING_OFFSETS
from .dvd_modulation import encode_recording_frames,nrzi
from .iso9660 import create_image
from .pipeline import write_json,sha256


def normalize_origin(cfg):
    """Absorb channel offset into effective radius and angle (exact model gauge)."""
    cfg=copy.deepcopy(cfg);d=cfg['disc'];p=d['track_pitch_um']*1e-3;b=d['channel_bit_length_nm']*1e-6
    r0=d['program_start_radius_mm']; effective=np.sqrt(r0*r0+p*b*d['channel_offset_bits']/np.pi)
    d['program_start_radius_mm']=float(effective)
    d['start_angle_deg']=float((d['start_angle_deg']+d['spiral_direction']*360*(effective-r0)/p)%360)
    d['channel_offset_bits']=0.0
    return cfg


def validate_pattern(cfg,pattern):
    inner,outer=pattern['annulus_inner_mm'],pattern['annulus_outer_mm']
    if not cfg['disc']['program_start_radius_mm']+.15<inner<outer<58:
        raise ValueError('Calibration annulus must lie inside the DVD-R program area')
    if pattern['ring_width_mm']<=0 or len(pattern['ring_centers_mm'])<3:
        raise ValueError('Need at least three rings with a positive width')
    rings=np.asarray(pattern['ring_centers_mm'],float)
    if np.any(np.diff(rings)<=pattern['ring_width_mm']*2):
        raise ValueError('Rings must be sorted and separated')
    if rings.min()-pattern['ring_width_mm']/2<=inner or rings.max()+pattern['ring_width_mm']/2>=outer:
        raise ValueError('Every ring must be fully inside the calibration annulus')
    if pattern['background_scrambled_byte']==pattern['feature_scrambled_byte']:
        raise ValueError('Calibration features require different data states')


def feature_mask(x,y,pattern):
    r=np.hypot(x,y);marked=np.zeros(np.shape(r),bool)
    for radius in pattern['ring_centers_mm']:
        marked|=np.abs(r-radius)<pattern['ring_width_mm']/2
    for spoke in pattern['spokes']:
        a=np.deg2rad(spoke['angle_deg']); along=x*np.cos(a)+y*np.sin(a); across=-x*np.sin(a)+y*np.cos(a)
        marked|=(along>0)&(np.abs(across)<spoke['width_mm']/2)
        # One/two side dots identify the arms independently of unknown rotation.
        for j in range(spoke['identification_dots']):
            dr=pattern['annulus_outer_mm']-.25-.35*j; da=a+np.deg2rad(1.2)
            marked|=(x-dr*np.cos(da))**2+(y-dr*np.sin(da))**2<.12**2
    return marked


def make_target(cfg,pattern):
    cfg=normalize_origin(cfg);validate_pattern(cfg,pattern);g=DvdSpiral(cfg['disc'])
    features=[]
    for i,r in enumerate(pattern['ring_centers_mm']):
        edge=r+pattern['ring_width_mm']/2
        features.append({'id':f'R{i+1:02d}','kind':'ring','label':f'R{i+1:02d}: OUTER edge {edge:g} mm',
                         'nominal_radius_mm':edge,'channel_index':float(g.sector_at_radius(edge)*BITS_PER_SECTOR),
                         'ring_center_mm':r,'edge':'outer'})
    for s in pattern['spokes']:
        features.append({'id':s['id'],'kind':'spoke','label':f"{s['id']}: {s['identification_dots']} side dot(s)",
                         'nominal_angle_deg':s['angle_deg'],'width_mm':s['width_mm'],
                         'identification_dots':s['identification_dots']})
    return {'schema':'dvd-calibration-v1','nominal_config':cfg,'pattern':pattern,'features':features,
            'measurement':{'disc_diameter_mm':pattern['disc_diameter_mm'],
                           'center_hole_diameter_mm':pattern['center_hole_diameter_mm'],
                           'ring_edge':'outer','trace_order':'inner to outer, continuously along one arm',
                           'scale_status':'Nominal diameter unless the user supplies a measured diameter',
                           'warning':'Affine ellipse correction assumes a near-normal photograph; it is not a full perspective/lens calibration.'}}


def preview(target,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=target['pattern']; limit=p['annulus_outer_mm']+5
    axis=np.linspace(-limit,limit,1800);x,y=np.meshgrid(axis,axis);r=np.hypot(x,y)
    z=np.ones(r.shape); inside=(r>=p['annulus_inner_mm'])&(r<=p['annulus_outer_mm'])
    z[inside]=.78;z[inside&feature_mask(x,y,p)]=.15
    z[r<p['center_hole_diameter_mm']/2]=np.nan
    fig,ax=plt.subplots(figsize=(9,9),layout='constrained')
    ax.imshow(z,origin='lower',cmap='gray',vmin=0,vmax=1,extent=[-limit,limit,-limit,limit])
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0,0),p['center_hole_diameter_mm']/2,facecolor='#eeeeee',edgecolor='#777777',linewidth=1))
    ax.text(0,0,f"{p['center_hole_diameter_mm']:g} mm\ncentre hole",ha='center',va='center',color='#555555',fontsize=10)
    ax.text(0,-15,f"Zoomed calibration band\nPhysical disc rim (r = {p['disc_diameter_mm']/2:g} mm) is outside this view",
            ha='center',va='center',color='#555555',fontsize=9)
    for f in target['features']:
        if f['kind']=='ring':
            rr=f['nominal_radius_mm'];a=np.deg2rad(45)
            ax.annotate(f['id'],xy=(rr*np.cos(a),rr*np.sin(a)),xytext=(rr*.86,rr*.87),fontsize=8,
                        arrowprops={'arrowstyle':'-','color':'#007c91'},color='#007c91')
        else:
            a=np.deg2rad(f['nominal_angle_deg']);rr=p['annulus_outer_mm']+2
            ax.text(rr*np.cos(a),rr*np.sin(a),f['id'],ha='center',va='center',color='#c64020',fontsize=12)
    ax.set(xlabel='Disc x (mm)',ylabel='Disc y (mm)',title='DVD-R calibration target - NOMINAL geometry\nR01..R09: outer ring edges; S01/S02: continuous spoke centres')
    ax.set_aspect('equal');fig.savefig(out/'target_preview.png',dpi=170);plt.close(fig)


def build_calibration(cfg,pattern_path,output):
    if cfg.get('format')!='DVD-R':raise ValueError('Calibration image currently supports DVD-R only')
    pattern=json.loads(Path(pattern_path).read_text(encoding='utf-8'))
    target=make_target(cfg,pattern);cfg=target['nominal_config'];g=DvdSpiral(cfg['disc'])
    out=Path(output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    first=max(32,int(np.floor(g.sector_at_radius(pattern['annulus_inner_mm']))))
    last=int(np.ceil(g.sector_at_radius(pattern['annulus_outer_mm'])))+1
    sectors=int(np.ceil((last+pattern['tail_sectors'])/16))*16
    if sectors>cfg['disc']['capacity_sectors']:raise ValueError('Calibration image exceeds media capacity')
    path=out/'dvd_calibration.iso';create_image(path,sectors,volume_id='DVD_R_CALIBRATION',
        description='DVD-R GEOMETRY CALIBRATION\r\nBurn the original ISO without rebuilding it.\r\nMeasure OUTER ring edges and continuous spoke centres; use companion target.json.\r\nHOLOGRAM.BIN is the fixed-extent calibration payload, not a viewable picture.\r\n')
    disk=np.memmap(path,mode='r+',dtype=np.uint8,shape=(sectors,2048))
    offsets=payload_channel_centers()[None,:];count=0;last_print=time.time()
    print(f'Generating calibration ISO: {sectors:,} sectors / {sectors*2048:,} bytes...',flush=True)
    for lo in range(first,last,pattern['chunk_sectors']):
        lbas=np.arange(lo,min(last,lo+pattern['chunk_sectors']),dtype=np.int64)
        x,y=g.coordinates(lbas[:,None],offsets);r=np.hypot(x,y)
        inside=(r>=pattern['annulus_inner_mm'])&(r<=pattern['annulus_outer_mm'])
        desired=np.where(feature_mask(x,y,pattern),pattern['feature_scrambled_byte'],pattern['background_scrambled_byte']).astype(np.uint8)
        payload=disk[lbas].copy();scrambled=scramble_payload(payload,lbas)
        scrambled[inside]=desired[inside];disk[lbas]=scramble_payload(scrambled,lbas)
        count+=int(inside.sum())
        if time.time()-last_print>20:
            print(f'  {lo-first:,}/{last-first:,} sectors',flush=True);last_print=time.time()
    disk.flush()
    # Verify actual image -> scrambler+ECC at independently recomputed sample positions.
    blocks=np.unique(np.linspace(first,last-17,7).astype(np.int64)//16)*16
    lbas=(blocks[:,None]+np.arange(16)).ravel()
    recorded=encode_ecc_blocks(disk[lbas].copy().reshape(-1,16,2048),blocks).reshape(-1,2366)
    x,y=g.coordinates(lbas[:,None],offsets);r=np.hypot(x,y)
    valid=(r>=pattern['annulus_inner_mm'])&(r<=pattern['annulus_outer_mm'])
    expected=np.where(feature_mask(x,y,pattern),pattern['feature_scrambled_byte'],pattern['background_scrambled_byte'])
    mismatches=int(np.count_nonzero(recorded[:,PAYLOAD_RECORDING_OFFSETS][valid]!=expected[valid]))
    del disk
    if mismatches:raise AssertionError('Calibration payload mapping disagrees with forward encoding')
    channel=encode_recording_frames(recorded[:16]);marks=nrzi(channel)
    run_lengths=np.diff(np.flatnonzero(channel));dsv=np.cumsum(1-2*marks.astype(np.int32))
    if np.any(~np.isin(run_lengths,np.r_[3:12,14])):raise AssertionError('Calibration channel run constraint failed')
    target['image']={'filename':path.name,'sectors':sectors,'first_pattern_lba':first,'last_pattern_lba':last-1,'sha256':sha256(path)}
    write_json(out/'target.json',target)
    with (out/'measurements_template.csv').open('w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerow(['feature_id','kind','click_index','x_mm','y_mm','radius_mm','theta_deg','angle_unwrapped_deg','sigma_mm'])
    preview(target,out)
    write_json(out/'manifest.json',{'purpose':'Geometric calibration only; no measurements have yet been taken',
        'sectors':sectors,'image_bytes':sectors*2048,'modified_payload_bytes':count,
        'forward_ecc_sample_blocks':blocks.tolist(),'checked_payload_bytes':int(valid.sum()),'payload_mismatches':mismatches,
        'channel_sample_bits':len(channel),'channel_run_lengths_T':np.unique(run_lengths).tolist(),
        'channel_max_abs_dsv':int(np.abs(dsv).max()),'elapsed_seconds':time.time()-start,
        'files_sha256':{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}})
    print(f'Created {path}; sample payload mismatches={mismatches}',flush=True)
    return target
