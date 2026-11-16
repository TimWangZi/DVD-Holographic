"""Generate host PCM, audit CIRC, and propagate a stated coarse mark model."""
import hashlib
import json
import math
import time
import wave
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from .geometry import Spiral
from .optics import binary_hologram, sample_grid, reflections, fresnel_roi, illumination
from . import encoding


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def export_wave(path,pcm,chunk=65536):
    # Canonical CIRC input is MSB-first per sample. RIFF PCM is LSB-first.
    with wave.open(str(path),'wb') as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(44100)
        for start in range(0,len(pcm),chunk):
            f.writeframesraw(pcm[start:start+chunk].reshape(-1,2)[:,::-1].copy().tobytes())


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def generate_pcm(cfg,x,y,mask):
    spiral=Spiral(cfg['disc']); patch=cfg['patch']
    outer=np.hypot(patch['center_radius_mm']+patch['size_mm'][1]/2,patch['size_mm'][0]/2)
    frames=int(np.ceil(spiral.frame_at_radius(outer)+cfg['disc']['tail_seconds']*7350+256))
    sectors=max(300,int(np.ceil(frames/98))); n=sectors*98
    if sectors>cfg['disc']['capacity_minutes']*60*75:
        raise ValueError('Generated track exceeds configured blank CD-R capacity')
    pcm=np.full((n,24),cfg['disc']['background_byte'],np.uint8)
    selected=spiral.patch_frames(patch)
    selected=selected[selected<n]
    positions=encoding.DATA_POSITIONS
    bits=44+17*positions+6.5
    xx,yy=spiral.local_coordinates(selected[:,None],bits[None,:],patch)
    inside=(np.abs(xx)<patch['size_mm'][0]/2)&(np.abs(yy)<patch['size_mm'][1]/2)
    palette=np.array(cfg['optimization']['palette_bytes'],dtype=np.uint8)
    # Minimum transition density leaves more recorded area after edge erosion.
    densities=encoding.EFM_TRANSITIONS[palette]
    low=int(palette[np.argmin(densities)]); high=int(palette[np.argmax(densities)])
    if low==high: raise ValueError('Palette needs at least two different EFM transition densities')
    sampled=sample_grid(mask,x,y,xx,yy)>0.5
    desired=np.where(sampled,low,high).astype(np.uint8)
    delays=encoding.DATA_DELAYS if cfg['optimization']['precompensate_circ'] else np.zeros(24,int)
    rows=selected[:,None]-delays[None,:]
    cols=np.broadcast_to(np.arange(24),rows.shape)
    valid=inside&(rows>=0)&(rows<n)
    pcm[rows[valid],cols[valid]]=desired[valid]
    return pcm,selected,desired,inside,{'sectors':sectors,'duration_seconds':sectors/75,
        'frames':n,'outer_program_radius_mm':float(np.hypot(*spiral.coordinates(n,0))),
        'palette_more_mark_area':low,'palette_less_mark_area':high,'controlled_symbols':int(valid.sum())}


def selected_circ(pcm,selected,chunk):
    result=np.empty((len(selected),32),np.uint8)
    # Exact overlap-save for finite CIRC delay lines, retaining only requested frames.
    history=int(max(encoding.DATA_DELAYS))+8
    # Parity paths can have longer history than any one systematic path.
    history=max(history,128)
    for start in range(0,len(pcm),chunk):
        end=min(start+chunk,len(pcm)); lo=np.searchsorted(selected,start); hi=np.searchsorted(selected,end)
        if lo==hi: continue
        origin=max(0,start-history)
        f2=encoding.encode_circ(pcm[origin:end])
        result[lo:hi]=f2[selected[lo:hi]-origin]
    return result


def coarse_mark_grid(cfg,x,y,selected,f2):
    spiral=Spiral(cfg['disc']); patch=cfg['patch']
    # Include sync and subcode cells in the optical envelope, not just the 32 F2 bytes.
    centers=np.r_[11.5,33.5,44+17*np.arange(32)+6.5]
    xx,yy=spiral.local_coordinates(selected[:,None],centers[None,:],patch)
    w,h=patch['size_mm']; keep=(np.abs(xx)<w/2+0.03)&(np.abs(yy)<h/2+0.003)
    # EFM word transition counts are exact. One mean merge transition per boundary
    # is NOT assumed: estimate from all legal merges for each actual adjacent pair.
    counts=encoding.EFM_TRANSITIONS[f2].astype(float)
    counts[:,:-1]+=encoding.mean_merge_transitions(f2[:,:-1],f2[:,1:])
    counts[:,-1]+=0.5  # last data symbol to next frame sync: coarse estimate
    subcounts=np.fromiter((sum(encoding._subcode_word(int(f),cfg['simulation']['subcode_absolute_offset_sectors'])) for f in selected),float,count=len(selected))
    density=np.column_stack([np.full(len(selected),sum(encoding.SYNC)/24),(subcounts+0.5)/17,counts/17])
    trim=cfg['media']['edge_trim_um']*1e-3/spiral.bit_mm
    area=np.clip(0.5-trim*density,0,1)*cfg['media']['mark_width_um']/cfg['disc']['track_pitch_um']
    tree=cKDTree(np.column_stack((xx[keep],yy[keep])))
    result=np.empty((len(y),len(x)),float)
    for start in range(0,len(y),128):
        gx,gy=np.meshgrid(x,y[start:start+128])
        _,ids=tree.query(np.column_stack((gx.ravel(),gy.ravel())),workers=1)
        result[start:start+128]=area[keep][ids].reshape(gx.shape)
    return result


def plot_outputs(out,cfg,x,y,mask,marks,verts,points,recon,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Image.fromarray(mask.astype(np.uint8)*255).save(out/'hologram.png')
    qnorm=(marks-marks.min())/max(np.ptp(marks),1e-15)
    Image.fromarray(np.uint8(qnorm*255)).save(out/'predicted_mark_envelope.png')
    fig=plt.figure(figsize=(13,8),layout='constrained')
    ax=fig.add_subplot(231,projection='3d')
    for i in range(4):
        for j in range(i+1,4):
            line=verts[[i,j]]; ax.plot(*line.T,color='#1487a5')
    ax.set(xlabel='Tangential x (mm)',ylabel='Radial y (mm)',zlabel='Distance z (mm)',title='Target: regular tetrahedron')
    encoding_label='RSPC / EFMPlus edge-area surrogate' if cfg.get('format')=='DVD-R' else 'CIRC / EFM edge-area surrogate'
    for index,(data,title) in enumerate([(mask,'Ideal binary hologram'),(marks,encoding_label)],2):
        ax=fig.add_subplot(2,3,index); ax.imshow(data,cmap='gray',origin='lower',extent=[x[0],x[-1],y[0],y[-1]],aspect='equal')
        ax.set(title=title,xlabel='Tangential mm',ylabel='Radial mm')
    common=max(float(recon['ideal'].max()),float(recon['encoded'].max()),1e-30)
    for index,key in enumerate(['ideal','encoded','zero_trim'],4):
        ax=fig.add_subplot(2,3,index)
        ax.imshow(10*np.log10(np.maximum(recon[key]/common,1e-6)),origin='lower',extent=recon['extent'],cmap='magma',vmin=-45,vmax=0)
        ax.set(title={'ideal':'Ideal two-state reconstruction','encoded':'Encoded surrogate reconstruction','zero_trim':'Zero edge trim: contrast collapses'}[key],xlabel='Tangential mm',ylabel='Radial mm')
    fig.suptitle(f"SIMULATION - uncalibrated {cfg.get('format','CD-R')} model; not a measured disc",fontsize=15)
    fig.savefig(out/'overview.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,len(recon['focus']),figsize=(5*len(recon['focus']),4),squeeze=False,layout='constrained')
    peak=max(max(float(im.max()),1e-30) for _,im in recon['focus'])
    for ax,(z,im) in zip(axes.flat,recon['focus']):
        ax.imshow(10*np.log10(np.maximum(im/peak,1e-6)),extent=recon['extent'],origin='lower',cmap='magma',vmin=-35,vmax=0)
        ax.set(title=f'Encoded surrogate z={z:g} mm',xlabel='Tangential mm',ylabel='Radial mm')
    fig.savefig(out/'focus_stack.png',dpi=150); plt.close(fig)


def run(cfg,output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True); started=time.time()
    print('Computing six-edge tetrahedron spherical-wave CGH...',flush=True)
    x,y,mask,verts,points,phases=binary_hologram(cfg)
    print('Projecting hologram through inverse CIRC data mapping...',flush=True)
    pcm,selected,desired,inside,metadata=generate_pcm(cfg,x,y,mask)
    print(f"Writing {metadata['duration_seconds']:.2f}s CD-DA WAV...",flush=True)
    export_wave(out/'tetrahedron.wav',pcm)
    (out/'tetrahedron.cue').write_text('REM CD-R CGH EXPERIMENT - DO NOT PLAY AS MUSIC\nFILE "tetrahedron.wav" WAVE\n  TRACK 01 AUDIO\n    INDEX 01 00:00:00\n',encoding='ascii')
    print('Forward CIRC including all parity bytes at aperture...',flush=True)
    f2=selected_circ(pcm,selected,cfg['simulation']['chunk_frames'])
    actual=f2[:,encoding.DATA_POSITIONS]
    mismatches=int(np.count_nonzero(actual[inside]!=desired[inside]))
    if cfg['optimization']['precompensate_circ'] and mismatches:
        raise AssertionError(f'Inverse/forward CIRC disagree at {mismatches} systematic symbols')
    print('Predicting mark-envelope and diffracted field...',flush=True)
    marks=coarse_mark_grid(cfg,x,y,selected,f2)
    ru,rm=reflections(cfg['media'],cfg['optics']['wavelength_nm'])
    fill=cfg['media']['mark_width_um']/cfg['disc']['track_pitch_um']
    # Ideal comparison uses the SAME weak edge-derived optical contrast, not a fictitious black/white disc.
    lo,hi=float(marks.min()),float(marks.max())
    ideal_q=np.where(mask,hi,lo)
    inc=illumination(cfg['optics'],x,y)
    n=cfg['simulation']['reconstruction_pixels']; span=cfg['simulation']['reconstruction_span_mm']
    xo=np.linspace(-span/2,span/2,n)+cfg['target']['offset_mm'][0]
    yo=np.linspace(-span/2,span/2,n)+cfg['target']['offset_mm'][1]
    recon={'extent':[float(xo[0]),float(xo[-1]),float(yo[0]),float(yo[-1])]}
    focus=[]; z=cfg['target']['distance_mm']; wl=cfg['optics']['wavelength_nm']
    for key,q in [('ideal',ideal_q),('encoded',marks),('zero_trim',np.full_like(marks,0.5*fill))]:
        field=(ru+(rm-ru)*q)*inc
        recon[key]=np.abs(fresnel_roi(field,x,y,wl,z,xo,yo))**2
        if key=='encoded':
            for dz in cfg['simulation']['focus_offsets_mm']:
                focus.append((z+dz,np.abs(fresnel_roi(field,x,y,wl,z+dz,xo,yo))**2))
    recon['focus']=focus
    # Channel audit on an actual generated waveform interval with real CIRC history.
    middle=int(selected[len(selected)//2]); ns=cfg['simulation']['channel_sample_frames']
    origin=max(0,middle-128); cf=encoding.encode_circ(pcm[origin:middle+ns])[middle-origin:]
    print('Encoding actual sample through EFM / merge / NRZI...',flush=True)
    initial=cfg['simulation']['channel_initial_mark']
    initial_dsv=cfg['simulation']['channel_initial_dsv']
    channel=encoding.encode_channel(cf,start_frame=middle,initial_mark=initial,initial_dsv=initial_dsv,
        absolute_offset=cfg['simulation']['subcode_absolute_offset_sectors'])
    bits=np.asarray(channel); nrzi=encoding.nrzi(bits,initial)
    runs=np.diff(np.flatnonzero(bits))
    mark_runs=encoding.mark_runs(bits,initial)
    trim_T=cfg['media']['edge_trim_um']*1e-3/Spiral(cfg['disc']).bit_mm
    eroded=np.maximum(mark_runs[mark_runs[:,2]==1,1]-2*trim_T,0).sum()/len(bits)
    np.savez_compressed(out/'channel_sample.npz',channel_bits=bits,marks=nrzi,mark_runs=mark_runs,f2=cf,start_frame=middle)
    audit={'frames':len(cf),'channel_bits':bits.size,'nrzi_mark_fraction':float(nrzi.mean()),
           'transition_density':float(bits.mean()),'eroded_mark_fraction':float(eroded),
           'min_run_T':int(runs.min()),'max_run_T':int(runs.max()),
           'start_frame':middle,'initial_mark':initial,'initial_dsv':initial_dsv}
    gx,gy=np.meshgrid(xo,yo)
    distances=np.full(gx.shape,np.inf)
    for i in range(4):
        for j in range(i+1,4):
            a=verts[i,:2]; b=verts[j,:2]; v=b-a
            t=np.clip(((gx-a[0])*v[0]+(gy-a[1])*v[1])/np.dot(v,v),0,1)
            distances=np.minimum(distances,np.hypot(gx-a[0]-t*v[0],gy-a[1]-t*v[1]))
    on=distances<0.10; off=(distances>0.20)&(distances<0.50)
    metadata.update({'systematic_circ_mismatches':mismatches,'shape_pixels':[len(y),len(x)],
        'land_r':[ru.real,ru.imag],'mark_r':[rm.real,rm.imag],
        'predicted_mark_area_range':[lo,hi], 'channel_audit':audit,
        'model':'CIRC exact reference; EFM exact table; merge policy model, not firmware capture. Full-aperture optics uses edge-area surrogate, not bit-resolved propagation.',
        'assumptions':['Unknown actual CD-R indices and mark morphology','Unknown firmware DSV/merge and write strategy','Nominal spiral pitch, velocity and frame origin, no physical calibration','No substrate aberration, vector grating diffraction or thermal model'],
        'validation_status':'Software checks only. Optical reconstruction on an ordinary writer is unverified.',
        'zero_trim_ROI_power_ratio':float(recon['zero_trim'].sum()/max(recon['encoded'].sum(),1e-30)),
        'encoded_edge_to_nearby_background_ratio':float(recon['encoded'][on].mean()/max(recon['encoded'][off].mean(),1e-30)),
        'diagnostic_edge_band_mm':0.10,'diagnostic_background_band_mm':[0.20,0.50],
        'elapsed_seconds':time.time()-started})
    plot_outputs(out,cfg,x,y,mask,marks,verts,points,recon,metadata)
    np.savez_compressed(out/'simulation.npz',x_mm=x,y_mm=y,binary=mask,mark_area=marks,vertices_mm=verts,points_mm=points,point_phases=phases,x_observation_mm=xo,y_observation_mm=yo,ideal_intensity=recon['ideal'],encoded_intensity=recon['encoded'],zero_trim_intensity=recon['zero_trim'])
    write_json(out/'config.json',cfg)
    metadata['files_sha256']={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}
    write_json(out/'manifest.json',metadata)
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)
    return metadata
