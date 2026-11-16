"""DVD-R host ISO -> inverse scrambling -> ECC/EFMPlus optical surrogate."""
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from . import dvd_encoding as enc
from . import dvd_modulation as mod
from .dvd_geometry import DvdSpiral, BITS_PER_SECTOR, payload_channel_centers, recording_channel_centers
from .iso9660 import create_image, PAYLOAD_START
from .optics import binary_hologram, sample_grid, reflections, illumination, fresnel_roi
from .pipeline import write_json,sha256,plot_outputs


def create_payload(cfg,path,x,y,mask):
    spiral=DvdSpiral(cfg['disc']); patch=cfg['patch']
    outer=np.hypot(patch['center_radius_mm']+patch['size_mm'][1]/2,patch['size_mm'][0]/2)
    sectors=int(np.ceil((spiral.sector_at_radius(outer)+cfg['disc']['tail_sectors'])/16))*16
    if sectors>cfg['disc']['capacity_sectors']: raise ValueError('Image exceeds single-layer DVD-R capacity')
    create_image(path,sectors)
    image=np.memmap(path,mode='r+',dtype=np.uint8,shape=(sectors,2048))
    if cfg['disc']['background_byte']:
        image[PAYLOAD_START:]=cfg['disc']['background_byte']
    selected=spiral.patch_sectors(patch)
    selected=selected[(selected>=PAYLOAD_START)&(selected<sectors)]
    palette=np.array(cfg['optimization']['palette_bytes'],np.uint8)
    densities=mod.EFMPLUS_TRANSITIONS[palette]
    more=int(palette[np.argmin(densities)]); less=int(palette[np.argmax(densities)])
    if more==less: raise ValueError('Palette must contain different EFMPlus transition densities')
    active=[]; controlled=0; mismatch=0; chunk=cfg['simulation']['sector_chunk']
    for start in range(0,len(selected),chunk):
        lbas=selected[start:start+chunk]
        xx,yy=spiral.local_coordinates(lbas[:,None],payload_channel_centers()[None,:],patch)
        inside=(np.abs(xx)<patch['size_mm'][0]/2)&(np.abs(yy)<patch['size_mm'][1]/2)
        if not inside.any(): continue
        desired=np.where(sample_grid(mask,x,y,xx,yy)>0.5,more,less).astype(np.uint8)
        payload=image[lbas].copy()
        if cfg['optimization']['precompensate_scrambler']:
            transformed=enc.scramble_payload(payload,lbas)
            transformed[inside]=desired[inside]
            payload=enc.scramble_payload(transformed,lbas)
        else:
            payload[inside]=desired[inside]
        image[lbas]=payload
        actual=enc.scramble_payload(payload,lbas)
        mismatch+=int(np.count_nonzero(actual[inside]!=desired[inside]))
        controlled+=int(inside.sum()); active.extend(lbas[inside.any(axis=1)].tolist())
    image.flush(); del image
    if cfg['optimization']['precompensate_scrambler'] and mismatch:
        raise AssertionError('DVD scrambler precompensation failed')
    return np.array(active,dtype=np.int64), {
        'sectors':sectors,'image_bytes':sectors*2048,'ecc_blocks':sectors//16,
        'outer_program_radius_mm':float(np.hypot(*spiral.coordinates(sectors))),
        'controlled_payload_bytes':controlled,'scrambler_mismatches':mismatch,
        'palette_more_mark_area':more,'palette_less_mark_area':less,
        'palette_mean_transitions':{str(int(b)):float(mod.EFMPLUS_TRANSITIONS[b]) for b in palette},
        'filesystem':'ISO9660: README.TXT and fixed-extent HOLOGRAM.BIN; burn image without rebuilding',
        'program_start_psn':enc.DATA_ZONE_START_PSN}


def forward_aperture(cfg,path,x,y,mask,active,metadata):
    image=np.memmap(path,mode='r',dtype=np.uint8,shape=(metadata['sectors'],2048))
    blocks=np.unique(active//16)*16; batch=cfg['simulation']['ecc_block_batch']
    spiral=DvdSpiral(cfg['disc']); patch=cfg['patch']
    total=np.zeros((len(y),len(x)),np.float32); count=np.zeros(total.shape,np.uint16)
    sync_centers=np.arange(26)*1488+15.5
    centers=np.r_[recording_channel_centers(),sync_centers]
    sync_density=np.array([np.mean([int(w).bit_count() for w in mod.SYNC[:,int(t),:].ravel()])/32 for t in mod.SYNC_SEQUENCE])
    trim=cfg['media']['edge_trim_um']*1e-3/spiral.bit_mm
    fill=cfg['media']['mark_width_um']/cfg['disc']['track_pitch_um']
    dx=x[1]-x[0]; dy=y[1]-y[0]; mismatches=0; checked=0
    started=time.time(); last=started
    for start in range(0,len(blocks),batch):
        bases=blocks[start:start+batch]; lbas=(bases[:,None]+np.arange(16)).ravel()
        payload=image[lbas].copy().reshape(-1,16,2048)
        recorded=enc.encode_ecc_blocks(payload,bases).reshape(-1,2366)
        # Check actual encoded systematic positions against the optical target.
        px,py=spiral.local_coordinates(lbas[:,None],payload_channel_centers()[None,:],patch)
        inside=(np.abs(px)<patch['size_mm'][0]/2)&(np.abs(py)<patch['size_mm'][1]/2)
        desired=np.where(sample_grid(mask,x,y,px,py)>0.5,
                         metadata['palette_more_mark_area'],metadata['palette_less_mark_area'])
        actual=recorded[:,enc.PAYLOAD_RECORDING_OFFSETS]
        mismatches+=int(np.count_nonzero(actual[inside]!=desired[inside])); checked+=int(inside.sum())
        density=np.column_stack([mod.EFMPLUS_TRANSITIONS[recorded]/16,np.broadcast_to(sync_density,(len(lbas),26))])
        area=np.clip(.5-trim*density,0,1)*fill
        xx,yy=spiral.local_coordinates(lbas[:,None],centers[None,:],patch)
        ix=np.floor((xx-x[0])/dx+.5).astype(np.int32); iy=np.floor((yy-y[0])/dy+.5).astype(np.int32)
        valid=(ix>=0)&(ix<len(x))&(iy>=0)&(iy<len(y))
        np.add.at(total,(iy[valid],ix[valid]),area[valid])
        np.add.at(count,(iy[valid],ix[valid]),1)
        if time.time()-last>20:
            print(f'  ECC/optical mapping: {min(start+batch,len(blocks))}/{len(blocks)} blocks',flush=True); last=time.time()
    del image
    if cfg['optimization']['precompensate_scrambler'] and mismatches:
        raise AssertionError(f'Encoded DVD systematic data differs from target at {mismatches} bytes')
    known=count>0; result=total/np.maximum(count,1)
    missing=~known
    if missing.any():
        indices=distance_transform_edt(missing,return_distances=False,return_indices=True)
        result[missing]=result[indices[0,missing],indices[1,missing]]
    metadata.update({'forward_ecc_blocks_checked':len(blocks),'encoded_payload_bytes_checked':checked,
        'encoded_payload_mismatches':mismatches,'raster_populated_fraction':float(known.mean()),
        'raster_gap_fill':'nearest measured-model cell; state-averaged EFMPlus envelope'})
    return result


def channel_audit(cfg,path,active,metadata,out):
    origin=int(active[len(active)//2]//16*16)
    sectors=cfg['simulation']['channel_sample_sectors']
    if sectors%16 or sectors<16: raise ValueError('Channel audit sectors must be a positive multiple of 16')
    with open(path,'rb') as f:
        f.seek(origin*2048); payload=np.frombuffer(f.read(sectors*2048),np.uint8).reshape(-1,16,2048)
    recording=enc.encode_ecc_blocks(payload,origin+np.arange(len(payload))*16).reshape(-1,2366)
    initial=cfg['simulation']['channel_initial_mark']
    bits,state=mod.encode_recording_frames(recording,initial_state=cfg['simulation']['channel_initial_state'],
        initial_mark=initial,initial_dsv=cfg['simulation']['channel_initial_dsv'],return_state=True)
    marks=mod.nrzi(bits,initial); runs=mod.mark_runs(bits,initial)
    intervals=np.diff(np.flatnonzero(bits))
    if np.any(~np.isin(intervals,np.r_[3:12,14])): raise AssertionError('Invalid EFMPlus run length')
    trim=cfg['media']['edge_trim_um']*1e-3/DvdSpiral(cfg['disc']).bit_mm
    eroded=np.maximum(runs[runs[:,2]==1,1]-2*trim,0).sum()/len(bits)
    dsv=cfg['simulation']['channel_initial_dsv']+np.cumsum(1-2*marks.astype(np.int32))
    np.savez_compressed(out/'channel_sample.npz',start_lba=origin,recording=recording,
                        channel_bits=bits,marks=marks,mark_runs=runs)
    return {'start_lba':origin,'sectors':sectors,'channel_bits':len(bits),'final_state':state,
            'nrzi_mark_fraction':float(marks.mean()),'eroded_mark_fraction':float(eroded),
            'transition_density':float(bits.mean()),'max_abs_dsv':int(np.abs(dsv).max()),
            'run_lengths_T':np.unique(intervals).tolist(),
            'note':'14T is the DVD sync run; ordinary data runs are 3T to 11T. This is a reference encoder, not a drive capture.'}


def optical_report(cfg,out,x,y,mask,marks,verts,points,phases,metadata):
    ru,rm=reflections(cfg['media'],cfg['optics']['wavelength_nm']); inc=illumination(cfg['optics'],x,y)
    n=cfg['simulation']['reconstruction_pixels']; span=cfg['simulation']['reconstruction_span_mm']
    xo=np.linspace(-span/2,span/2,n)+cfg['target']['offset_mm'][0]
    yo=np.linspace(-span/2,span/2,n)+cfg['target']['offset_mm'][1]
    z=cfg['target']['distance_mm']; wl=cfg['optics']['wavelength_nm']
    lo,hi=float(marks.min()),float(marks.max()); fill=cfg['media']['mark_width_um']/cfg['disc']['track_pitch_um']
    recon={'extent':[float(xo[0]),float(xo[-1]),float(yo[0]),float(yo[-1])],'focus':[]}
    for key,q in [('ideal',np.where(mask,hi,lo)),('encoded',marks),('zero_trim',np.full_like(marks,.5*fill))]:
        field=(ru+(rm-ru)*q)*inc
        recon[key]=np.abs(fresnel_roi(field,x,y,wl,z,xo,yo))**2
        if key=='encoded':
            for dz in cfg['simulation']['focus_offsets_mm']:
                recon['focus'].append((z+dz,np.abs(fresnel_roi(field,x,y,wl,z+dz,xo,yo))**2))
    gx,gy=np.meshgrid(xo,yo); distances=np.full(gx.shape,np.inf)
    for i in range(4):
        for j in range(i+1,4):
            a=verts[i,:2]; v=verts[j,:2]-a
            t=np.clip(((gx-a[0])*v[0]+(gy-a[1])*v[1])/np.dot(v,v),0,1)
            distances=np.minimum(distances,np.hypot(gx-a[0]-t*v[0],gy-a[1]-t*v[1]))
    on=distances<.10; off=(distances>.20)&(distances<.50)
    metadata.update({'land_r':[ru.real,ru.imag],'mark_r':[rm.real,rm.imag],
        'predicted_mark_area_range':[lo,hi], 'shape_pixels':list(mask.shape),
        'zero_trim_ROI_power_ratio':float(recon['zero_trim'].sum()/max(recon['encoded'].sum(),1e-30)),
        'encoded_edge_to_nearby_background_ratio':float(recon['encoded'][on].mean()/max(recon['encoded'][off].mean(),1e-30)),
        'diagnostic_edge_band_mm':.1,'diagnostic_background_band_mm':[.2,.5]})
    plot_outputs(out,cfg,x,y,mask,marks,verts,points,recon,metadata)
    np.savez_compressed(out/'simulation.npz',x_mm=x,y_mm=y,binary=mask,mark_area=marks,
        vertices_mm=verts,points_mm=points,point_phases=phases,x_observation_mm=xo,y_observation_mm=yo,
        ideal_intensity=recon['ideal'],encoded_intensity=recon['encoded'],zero_trim_intensity=recon['zero_trim'])


def run(cfg,output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True); started=time.time()
    print('Computing DVD-R tetrahedron spherical-wave hologram...',flush=True)
    x,y,mask,verts,points,phases=binary_hologram(cfg)
    print('Writing ISO9660 image and compensating DVD payload scrambler...',flush=True)
    path=out/'tetrahedron_dvd.iso'
    active,metadata=create_payload(cfg,path,x,y,mask)
    print(f"Image: {metadata['image_bytes']:,} bytes, {metadata['sectors']} sectors; forward ECC...",flush=True)
    marks=forward_aperture(cfg,path,x,y,mask,active,metadata)
    print('Encoding full EFMPlus audit sample including DVD sync...',flush=True)
    metadata['channel_audit']=channel_audit(cfg,path,active,metadata,out)
    print('Propagating optical fields and writing previews...',flush=True)
    optical_report(cfg,out,x,y,mask,marks,verts,points,phases,metadata)
    metadata.update({'format':'single-layer DVD-R, ISO9660, 2048-byte logical sectors',
        'model':'Actual ISO bytes -> ID/IED/EDC -> scrambler -> PO/PI ECC -> recording interleave. Full aperture uses state-averaged EFMPlus transition density and edge-area surrogate. Full channel sample uses reference two-stream DSV.',
        'assumptions':['Unmeasured dye/metal optical constants and mark-edge trim','Nominal pitch, bit length and program-area origin','State average is not the firmware-selected per-symbol EFMPlus stream','No substrate aberration, vector groove scattering or thermal write model'],
        'validation_status':'Software verified reference model; physical DVD-R holographic reconstruction unverified.',
        'elapsed_seconds':time.time()-started})
    write_json(out/'config.json',cfg)
    metadata['files_sha256']={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}
    write_json(out/'manifest.json',metadata)
    print(f"Done: {path}\nEncoded payload mismatches: {metadata['encoded_payload_mismatches']}\n"
          f"Edge/background ratio (simulation): {metadata['encoded_edge_to_nearby_background_ratio']:.3f}\n"
          f"Elapsed: {metadata['elapsed_seconds']:.1f}s",flush=True)
    return metadata
