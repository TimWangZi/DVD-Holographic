"""Fit DVD-R geometry from known-clock rings and continuously traced spokes.

The free channel-clock offset is gauge-equivalent to the radius/angle origin and
is fixed to zero. Photo traces use their supplied winding numbers: this module
does not turn sparse modulo-angle measurements into a claimed unique solution.
"""
import copy
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


def _nominal(config):
    d=config['disc']; p=d['track_pitch_um']*1e-3; b=d['channel_bit_length_nm']*1e-6
    B=p*b/np.pi
    r=np.sqrt(d['program_start_radius_mm']**2+B*d['channel_offset_bits'])
    angle=d['start_angle_deg']+d['spiral_direction']*360*(r-d['program_start_radius_mm'])/p
    return r,p,b,B,angle


def _read(target_path,measurements_path):
    target=json.loads(Path(target_path).read_text(encoding='utf-8'))
    if target.get('schema')!='dvd-calibration-v1':
        raise ValueError('Expected dvd-calibration-v1 target manifest')
    cfg=target['nominal_config']
    if cfg.get('format')!='DVD-R': raise ValueError('This fitter only supports DVD-R')
    features={f['id']:f for f in target['features']}
    rings=[]; traces=[]
    default_sigma=float(target.get('measurement',{}).get('sigma_mm',0.025))
    if default_sigma<=0: raise ValueError('Measurement uncertainty must be positive')
    with Path(measurements_path).open(encoding='utf-8-sig',newline='') as f:
        for line,row in enumerate(csv.DictReader(f),2):
            fid=row.get('feature_id','')
            if fid not in features: raise ValueError(f'Line {line}: unknown feature {fid!r}')
            feature=features[fid]; kind=feature['kind']
            if kind not in ('ring','spoke'): continue
            if row.get('kind',kind)!=kind: raise ValueError(f'Line {line}: feature kind mismatch')
            try:
                point={'id':fid,'x':float(row['x_mm']),'y':float(row['y_mm']),
                       'index':int(row['click_index']),
                       'sigma':float(row.get('sigma_mm') or default_sigma)}
                if kind=='ring': point['clock']=float(feature['channel_index'])
                else:
                    point['angle']=float(row['angle_unwrapped_deg'])
                    point['nominal_angle']=float(feature['nominal_angle_deg'])
            except (KeyError,ValueError) as error:
                raise ValueError(f'Line {line}: invalid measurement; spokes need an explicit unwrapped angle') from error
            if not all(np.isfinite(v) for k,v in point.items() if k!='id') or point['sigma']<=0:
                raise ValueError(f'Line {line}: measurements must be finite and sigma_mm positive')
            (rings if kind=='ring' else traces).append(point)
    ids=sorted({p['id'] for p in rings})
    if len(ids)<2: raise ValueError('At least two separated known-clock rings are required')
    for fid in ids:
        pts=[p for p in rings if p['id']==fid]
        if len(pts)<6: raise ValueError(f'Ring {fid} needs at least six points around its circumference')
        angles=np.sort(np.arctan2([p['y'] for p in pts],[p['x'] for p in pts]))
        if np.diff(np.r_[angles,angles[0]+2*np.pi]).max()>np.pi:
            raise ValueError(f'Ring {fid} points must cover more than a semicircle')
    means=[np.mean([np.hypot(p['x'],p['y']) for p in rings if p['id']==fid]) for fid in ids]
    if np.ptp(means)<0.5: raise ValueError('Radial anchors must span at least 0.5 mm')
    return target,cfg,rings,traces


def _arrays(points,keys):
    return tuple(np.asarray([p[k] for p in points],float) for k in keys)


def _radial_fit(cfg,rings):
    rn,pn,bn,Bn,_=_nominal(cfg)
    x,y,N,sigma=_arrays(rings,['x','y','clock','sigma'])
    scale=max(float(N.max()),1.)
    coeff=np.linalg.lstsq(np.column_stack([np.ones(len(N)),N/scale]),x*x+y*y,rcond=None)[0]
    A,B=coeff[0],coeff[1]/scale
    if A<=0 or B<=0: raise ValueError('Ring labels/radii imply a nonphysical radial geometry')
    initial=np.array([np.sqrt(A),np.log(B/Bn),0.,0.])
    lower=np.array([rn-1.,np.log(.8),-1.,-1.]); upper=np.array([rn+1.,np.log(1.2),1.,1.])
    if np.any(initial<=lower) or np.any(initial>=upper):
        raise ValueError('Ring fit is far outside the nominal geometry; check scale, labels and disc centre')
    def residual(z):
        r0,logB,cx,cy=z
        expected=np.sqrt(r0*r0+Bn*np.exp(logB)*N)
        return (np.hypot(x-cx,y-cy)-expected)/sigma
    result=least_squares(residual,initial,bounds=(lower,upper),x_scale='jac',loss='soft_l1',
                         ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=1000)
    if not result.success: raise ValueError(f'Radial fit did not converge: {result.message}')
    if np.any(result.active_mask): raise ValueError('Radial fit reached parameter bounds; check scale, ring labels and centre')
    return result,residual,(lower,upper)


def _prepare_traces(traces,cx,cy,nominal_direction,direction):
    ids=sorted({p['id'] for p in traces})
    if not ids: raise ValueError('Refine mode needs at least one continuously traced spoke')
    groups=[]
    for fid in ids:
        pts=sorted([dict(p) for p in traces if p['id']==fid],key=lambda p:p['index'])
        if len(pts)<8: raise ValueError(f'Spoke {fid} needs at least eight ordered trace points')
        if len({p['index'] for p in pts})!=len(pts): raise ValueError(f'Duplicate click_index in {fid}')
        radii=np.hypot([p['x']-cx for p in pts],[p['y']-cy for p in pts])
        if np.ptp(radii)<0.5: raise ValueError(f'Spoke {fid} must span at least 0.5 mm radially')
        dr=np.diff(radii)
        if not (np.all(dr>0) or np.all(dr<0)):
            raise ValueError(f'Spoke {fid} must be traced in strictly radial order')
        steps=np.diff([p['angle'] for p in pts])
        if np.max(np.abs(steps))>=150:
            raise ValueError(f'Spoke {fid} has a >=150 degree jump; trace it more densely without skipping turns')
        groups.append((pts,radii))
    lo=max(float(r.min()) for _,r in groups); hi=min(float(r.max()) for _,r in groups)
    if hi<=lo: raise ValueError('Spoke traces need overlapping radial coverage')
    reference=(lo+hi)/2
    phase_at_reference=[]
    for pts,radii in groups:
        raw=np.degrees(np.arctan2([p['y']-cy for p in pts],[p['x']-cx for p in pts]))
        unwrapped=raw+360*np.rint((np.array([p['angle'] for p in pts])-raw)/360)
        order=np.argsort(radii)
        phase_at_reference.append(float(np.interp(reference,radii[order],unwrapped[order])))
    nominal_phases=np.array([pts[0]['nominal_angle'] for pts,_ in groups])
    phase_at_reference=np.array(phase_at_reference)
    scores={}
    for ratio in (1,-1):
        phase=phase_at_reference-ratio*nominal_phases
        mean=np.degrees(np.angle(np.exp(1j*np.deg2rad(phase)).mean()))
        error=(phase-mean+180)%360-180
        scores[ratio]=float(np.sqrt(np.mean(error**2)))
    if direction=='auto':
        if len(groups)<2:
            raise ValueError('Automatic direction requires two asymmetric spokes; explicitly set direction for a single trace')
        ratio=min(scores,key=scores.get)
        if scores[ratio]>5 or scores[-ratio]-scores[ratio]<10:
            raise ValueError('Spoke geometry does not uniquely identify direction; check spoke labels or trace two asymmetric spokes')
        actual_direction=nominal_direction*ratio
    else:
        actual_direction=int(direction); ratio=actual_direction*nominal_direction
        if len(groups)>=2 and scores[ratio]>5:
            raise ValueError('Requested direction disagrees with measured spoke separation; check direction and labels')
    direction_report={'selection':'inferred_from_asymmetric_spokes' if direction=='auto' else 'explicit_user_value',
                      'nominal_direction':nominal_direction,'actual_direction':actual_direction,'phase_ratio':ratio,
                      'circular_phase_rms_deg':scores[ratio],
                      'candidate_phase_rms_deg':{str(nominal_direction*q):value for q,value in scores.items()}}
    aligned=[]; branch_reference=None
    for pts,radii in groups:
        raw=np.degrees(np.arctan2([p['y']-cy for p in pts],[p['x']-cx for p in pts]))
        unwrapped=raw+360*np.rint((np.array([p['angle'] for p in pts])-raw)/360)
        phase=unwrapped-ratio*np.array([p['nominal_angle'] for p in pts])
        order=np.argsort(radii); at_ref=float(np.interp(reference,radii[order],phase[order]))
        if branch_reference is None: branch_reference=at_ref
        branch=360*round((branch_reference-at_ref)/360)
        for point in pts:
            point['angle']+=branch
            point['nominal_angle']*=ratio
            aligned.append(point)
    return aligned,reference,actual_direction,direction_report


def _trace_values(z,points,cfg,reference,direction):
    rn,pn,bn,Bn,_=_nominal(cfg)
    r0,logB,u,beta,cx,cy=z; B=Bn*np.exp(logB)
    x,y,provided,nominal,sigma=_arrays(points,['x','y','angle','nominal_angle','sigma'])
    r=np.hypot(x-cx,y-cy); raw=np.degrees(np.arctan2(y-cy,x-cx))
    observed=raw+360*np.rint((provided-raw)/360)-nominal
    source_r=np.sqrt(rn*rn+(Bn/B)*(r*r-r0*r0))
    source_ref=np.sqrt(rn*rn+(Bn/B)*(reference*reference-r0*r0))
    # Cancel the large common number of turns before adding the small pitch correction.
    predicted=beta+direction*360*((r-reference)-(source_r-source_ref)+u*(r-reference))/pn
    derivative=direction*360*((1+u)-(Bn/B)*r/source_r)/pn
    angular_sigma=np.hypot(np.degrees(sigma/r),derivative*sigma)
    return r,observed,predicted,angular_sigma


def _refine(cfg,rings,traces,coarse,reference,direction,train=None,initial=None):
    rn,pn,bn,Bn,_=_nominal(cfg)
    r0,logB,cx,cy=coarse.x
    if initial is None:
        z=np.array([r0,logB,0.,0.,cx,cy])
        r,observed,geometric,_=_trace_values(z,traces,cfg,reference,direction)
        slope=direction*360*(r-reference)/pn
        beta,u=np.linalg.lstsq(np.column_stack([np.ones(len(r)),slope]),observed-geometric,rcond=None)[0]
        initial=np.array([r0,logB,u,beta,cx,cy])
    lower=np.array([rn-1.,np.log(.8),-.15,-np.inf,-1.,-1.])
    upper=np.array([rn+1.,np.log(1.2),.15,np.inf,1.,1.])
    if np.any(initial<=lower) or np.any(initial>=upper):
        raise ValueError('Trace implies implausible geometry; check spoke identity, scale and winding')
    xr,yr,N,sr=_arrays(rings,['x','y','clock','sigma'])
    fixed_sigma=_trace_values(initial,traces,cfg,reference,direction)[3]
    if train is None: train=np.ones(len(traces),bool)
    def residual(z):
        a,logb,u,beta,cx,cy=z
        radial=(np.hypot(xr-cx,yr-cy)-np.sqrt(a*a+Bn*np.exp(logb)*N))/sr
        _,observed,predicted,_=_trace_values(z,traces,cfg,reference,direction)
        return np.r_[radial,((observed-predicted)/fixed_sigma)[train]]
    result=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',x_scale='jac',
                         ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=1500)
    if not result.success: raise ValueError(f'Joint geometry fit did not converge: {result.message}')
    if np.any(result.active_mask): raise ValueError('Joint fit reached parameter bounds; check scale, spoke identity and winding')
    return result,residual


def _uncertainty(result,residual):
    # Supplied measurement sigma is a lower variance scale even for synthetic/exact residuals.
    z=result.x; steps=np.maximum(np.abs(z)*1e-6,1e-6)
    jac=np.column_stack([(residual(z+np.eye(len(z))[i]*steps[i])-residual(z-np.eye(len(z))[i]*steps[i]))/(2*steps[i]) for i in range(len(z))])
    u,s,vt=np.linalg.svd(jac,full_matrices=False)
    rank=int(np.count_nonzero(s>s[0]*np.finfo(float).eps*max(jac.shape)))
    if rank<len(z): raise ValueError('Measurements do not identify the selected fit parameters')
    dof=max(1,len(result.fun)-len(z)); scale=max(1.,float(np.dot(result.fun,result.fun)/dof))
    covariance=(vt.T*(1/s**2))@vt*scale
    return covariance,rank,float(s[0]/s[-1]),scale


def fit_calibration(target_path,measurements_path,output,mode='refine',direction='auto'):
    if mode not in ('coarse','refine'): raise ValueError('mode must be coarse or refine')
    if str(direction) not in ('auto','1','-1'): raise ValueError('direction must be auto, 1 or -1')
    direction=str(direction)
    target,cfg,rings,traces=_read(target_path,measurements_path)
    rn,pn,bn,Bn,nominal_angle=_nominal(cfg)
    coarse,radial_fun,_=_radial_fit(cfg,rings)
    heldout=None; reference=None; aligned=[]
    if mode=='refine':
        aligned,reference,actual_direction,direction_report=_prepare_traces(traces,coarse.x[2],coarse.x[3],cfg['disc']['spiral_direction'],direction)
        full,residual=_refine(cfg,rings,aligned,coarse,reference,actual_direction)
        keep=np.ones(len(aligned),bool)
        for fid in sorted({p['id'] for p in aligned}):
            indices=[i for i,p in enumerate(aligned) if p['id']==fid]
            keep[np.array(indices[2:-1:4],dtype=int)]=False
        trained,_=_refine(cfg,rings,aligned,coarse,reference,actual_direction,train=keep,initial=full.x)
        rr,obs,pred,_=_trace_values(trained.x,aligned,cfg,reference,actual_direction)
        error=(obs-pred)*np.pi/180*rr
        heldout={'point_count':int((~keep).sum()),'tangential_rms_um':float(np.sqrt(np.mean(error[~keep]**2))*1000),
                 'tangential_max_abs_um':float(np.max(np.abs(error[~keep]))*1000),
                 'method':'Every fourth interior trace point held out; all rings retained. Final configuration fitted to all points.'}
        r0,logB,u,beta,cx,cy=full.x; B=Bn*np.exp(logB); pitch=pn/(1+u); bit=np.pi*B/pitch
        rr,obs,pred,_=_trace_values(full.x,aligned,cfg,reference,actual_direction)
        angle_error=obs-pred; tangent=angle_error*np.pi/180*rr
        source_ref=np.sqrt(rn*rn+(Bn/B)*(reference*reference-r0*r0))
        ratio=actual_direction*cfg['disc']['spiral_direction']
        photo_angle=(beta+ratio*nominal_angle-actual_direction*360*((reference-r0)/pitch-(source_ref-rn)/pn))%360
    else:
        full,residual=coarse,radial_fun
        r0,logB,cx,cy=full.x; B=Bn*np.exp(logB); bit=bn; pitch=np.pi*B/bit
        photo_angle=None; tangent=np.array([]); angle_error=np.array([])
        actual_direction=cfg['disc']['spiral_direction'] if direction=='auto' else int(direction)
        direction_report={'selection':'not_estimated_coarse_preserves_nominal' if direction=='auto' else 'explicit_user_value',
                          'nominal_direction':cfg['disc']['spiral_direction'],'actual_direction':actual_direction}
    covariance,rank,condition,scale=_uncertainty(full,residual)
    sigma_r0=float(np.sqrt(covariance[0,0]))
    if mode=='refine':
        grad_p=np.array([0.,0.,-pn/(1+u)**2,0.,0.,0.])
        grad_b=np.array([0.,bit,bit/(1+u),0.,0.,0.])
        sigma_p=float(np.sqrt(grad_p@covariance@grad_p)*1000)
        sigma_b=float(np.sqrt(grad_b@covariance@grad_b)*1e6)
    else:
        sigma_p=float(pitch*np.sqrt(covariance[1,1])*1000); sigma_b=None
    xr,yr,N,sr=_arrays(rings,['x','y','clock','sigma'])
    radial_error=np.hypot(xr-cx,yr-cy)-np.sqrt(r0*r0+B*N)
    warnings=['Photo scale, lens distortion and ellipse rectification are not included in conditional covariance.',
              'A fitted residual is not a guarantee of micron-scale hologram accuracy on another blank disc.',
              'Channel offset fixed to zero; radius is an effective origin, not a separately measured physical start radius.']
    if mode=='coarse': warnings.append('Bit length fixed to nominal: pitch is conditional. Coarse fit does not correct angular twist.')
    else:
        warnings.extend(['Trace winding must have been followed continuously; missed full turns cannot be excluded from these samples alone.',
                         'Photo orientation is not transferable to a later burn; nominal start_angle_deg is preserved in the exported configuration.'])
    if condition>1e8: warnings.append('Fit Jacobian is poorly conditioned; individual physical parameters are weakly separated.')
    report={'schema':'dvd-calibration-fit-v1','mode':mode,'status':'experimental_geometry_fit_not_hologram_certification',
        'model':'Known-clock rings plus unwrapped nominal-spoke residual twist; common photo-centre correction; channel-offset gauge fixed to zero.',
        'trace_reference_radius_mm':reference,
        'ring_points':len(rings),'spoke_points':len(aligned),'parameters':{
            'effective_program_start_radius_mm':float(r0),'track_pitch_um':float(pitch*1000),
            'channel_bit_length_nm':float(bit*1e6),'channel_offset_bits':0.,
            'spiral_direction':actual_direction,
            'photo_center_correction_mm':[float(cx),float(cy)],'photo_start_angle_deg':photo_angle,
            'exported_start_angle_deg':cfg['disc']['start_angle_deg']},
        'conditional_standard_errors':{'effective_radius_mm':sigma_r0,'track_pitch_um':sigma_p,
            'channel_bit_length_nm':sigma_b,'bit_length_fixed':mode=='coarse'},
        'residuals':{'ring_radial_rms_um':float(np.sqrt(np.mean(radial_error**2))*1000),
            'ring_radial_max_abs_um':float(np.max(np.abs(radial_error))*1000),
            'trace_tangential_rms_um':float(np.sqrt(np.mean(tangent**2))*1000) if len(tangent) else None,
            'trace_tangential_max_abs_um':float(np.max(np.abs(tangent))*1000) if len(tangent) else None},
        'held_out_validation':heldout,'direction_inference':direction_report,'jacobian_rank':rank,'jacobian_condition_number':condition,
        'variance_scale':scale,'winding_ambiguity_excluded':False,'warnings':warnings,
        'source_target':str(Path(target_path).resolve()),'source_measurements':str(Path(measurements_path).resolve())}
    # A quantitative registration reference for the CURRENT hologram, not a
    # certificate of camera calibration or successful optical reconstruction.
    from .optics import tetrahedron
    vertices,_,_=tetrahedron(cfg['target'])
    extent=np.max(np.abs(vertices[:,:2]),axis=0)+np.asarray(cfg['patch']['size_mm'])/2
    incident=np.abs(np.sin(np.deg2rad(cfg['optics']['illumination_angles_deg'])))
    frequency=float(np.linalg.norm(extent/vertices[:,2].min()+incident)/(cfg['optics']['wavelength_nm']*1e-6))
    period_um=1000/frequency; budget_um=period_um/8
    validation_um=None if heldout is None else float(np.hypot(report['residuals']['ring_radial_rms_um'],heldout['tangential_rms_um']))
    report['hologram_registration_reference']={
        'nominal_min_fringe_period_um':period_um,
        'pi_over_4_phase_error_position_budget_um':budget_um,
        'combined_ring_and_heldout_rms_um':validation_um,
        'within_heuristic_budget':None if validation_um is None else validation_um<=budget_um,
        'meaning':'Conservative point-source spatial-frequency bound; period/8 corresponds to a pi/4 phase error. Training/held-out residuals omit camera systematic errors and disc-to-disc drift, so passing is not optical certification.'}
    updated=copy.deepcopy(cfg)
    updated['disc'].update({'program_start_radius_mm':float(r0),'track_pitch_um':float(pitch*1000),
                           'channel_bit_length_nm':float(bit*1e6),'channel_offset_bits':0.,'spiral_direction':actual_direction})
    updated['calibration']={'mode':mode,'status':report['status'],'measurements':report['source_measurements'],
                            'orientation_transferable':False,'material_parameters_measured':False}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('calibrated_config.json',updated),('fit_report.json',report)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    _plot(out,rings,radial_error,aligned,tangent,report,cfg,full.x,reference)
    return report


def _plot(out,rings,radial_error,traces,tangent,report,cfg,z,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    ax=axes[0,0]
    ax.scatter([p['clock']/38688 for p in rings],radial_error*1000,s=14)
    ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Known LBA-equivalent clock',ylabel='Radial residual (um)',title='Ring anchors')
    if traces:
        r,obs,pred,_=_trace_values(z,traces,cfg,reference,report['parameters']['spiral_direction'])
        for fid in sorted({p['id'] for p in traces}):
            k=np.array([p['id']==fid for p in traces]); order=np.argsort(r[k])
            axes[0,1].scatter(r[k],obs[k],s=16,label=fid)
            axes[0,1].plot(r[k][order],pred[k][order],lw=1)
        axes[0,1].legend();axes[0,1].set(xlabel='Observed radius (mm)',ylabel='Residual twist (deg)',title='Unwrapped spoke trace and fit')
        axes[1,0].scatter(r,tangent*1000,s=14);axes[1,0].axhline(0,color='gray',lw=.7)
        axes[1,0].set(xlabel='Observed radius (mm)',ylabel='Tangential residual (um)',title='Trace registration residual')
    else:
        for ax in (axes[0,1],axes[1,0]): ax.axis('off')
        axes[0,1].text(.05,.5,'Coarse radial fit only\nBit length held fixed; no angular validation',transform=axes[0,1].transAxes)
    p=report['parameters']; axes[1,1].axis('off')
    text=(f"Mode: {report['mode']}\nEffective radius: {p['effective_program_start_radius_mm']:.7f} mm\n"
          f"Pitch: {p['track_pitch_um']:.7f} um\nBit length: {p['channel_bit_length_nm']:.7f} nm\n"
          f"Ring RMS: {report['residuals']['ring_radial_rms_um']:.3f} um\n\n"
          'Conditional fit precision only.\nPhoto/systematic errors remain.\nNot a physical hologram validation.')
    axes[1,1].text(.02,.95,text,va='top',transform=axes[1,1].transAxes,fontsize=11)
    fig.suptitle('DVD-R geometry calibration residuals')
    fig.savefig(out/'fit_residuals.png',dpi=150);plt.close(fig)
