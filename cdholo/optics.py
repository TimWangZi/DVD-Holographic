"""Scalar reflection model, spherical point-source CGH and Fresnel propagation."""
import itertools
import numpy as np
from scipy.ndimage import map_coordinates


def tetrahedron(cfg):
    verts = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
    verts *= cfg['edge_mm']/(2*np.sqrt(2))
    for axis, deg in enumerate(cfg['rotation_deg']):
        a = np.deg2rad(deg); c,s=np.cos(a),np.sin(a)
        if axis==1: s=-s  # right-handed Y rotation in the (x,z) submatrix
        ids=[i for i in range(3) if i!=axis]
        rot=np.eye(3); rot[np.ix_(ids,ids)]=[[c,-s],[s,c]]
        verts=verts@rot.T
    verts += np.array([*cfg['offset_mm'],cfg['distance_mm']])
    points=[]
    for i,j in itertools.combinations(range(4),2):
        points.extend(verts[i]+(verts[j]-verts[i])*t for t in np.linspace(0,1,cfg['points_per_edge'],endpoint=False))
    points=np.array(points)
    rng=np.random.default_rng(cfg['phase_seed'])
    phases=rng.uniform(0,2*np.pi,len(points)) if cfg['random_point_phase'] else np.zeros(len(points))
    return verts,points,phases


def film_reflection(n0, layers, ns, wavelength_mm, angle_deg, polarization):
    """Passive indices n+ik, exp(-i wt), oblique thin-film admittance recursion."""
    s = n0*np.sin(np.deg2rad(angle_deg))
    def q(n):
        ct=np.sqrt(1-(s/n)**2+0j)
        return (n*ct if polarization=='s' else ct/n), n*ct
    ys,_=q(ns); y=ys
    for layer in reversed(layers):
        n=complex(*layer['n_k']); eta,nc=q(n)
        delta=2*np.pi*nc*layer['thickness_nm']*1e-6/wavelength_mm
        t=np.tan(delta)
        y=eta*(y-1j*eta*t)/(eta-1j*y*t)
    y0,_=q(n0)
    return (y0-y)/(y0+y)


def reflections(cfg, wavelength_nm):
    if cfg['reflection_mode']=='direct':
        return complex(*cfg['land_r']),complex(*cfg['mark_r'])
    if cfg['reflection_mode']!='thin_film':
        raise ValueError('reflection_mode must be direct or thin_film')
    args=(cfg['substrate_n'],complex(*cfg['reflector_n_k']),wavelength_nm*1e-6,
          cfg['internal_incidence_deg'],cfg['polarization'])
    ru=film_reflection(args[0],cfg['land_layers'],*args[1:])
    rm=film_reflection(args[0],cfg['mark_layers'],*args[1:])
    return ru,rm


def grid(cfg):
    w,h=cfg['patch']['size_mm']; dx,dy=np.array(cfg['simulation']['grid_step_um'])*1e-3
    nx=int(np.ceil(w/dx)); ny=int(np.ceil(h/dy))
    # Use exact requested pitch; last pixel center stays inside the aperture.
    x=(np.arange(nx)-(nx-1)/2)*dx; y=(np.arange(ny)-(ny-1)/2)*dy
    return x,y


def object_wave(x,y,points,phases,wavelength_nm):
    xx,yy=np.meshgrid(x,y); field=np.zeros(xx.shape,complex)
    k=2*np.pi/(wavelength_nm*1e-6)
    for (px,py,pz),phi in zip(points,phases):
        # Subtract axial distance to avoid enormous common phase; point phase defines it.
        d=np.sqrt((xx-px)**2+(yy-py)**2+pz*pz)
        field += np.exp(-1j*k*(d-pz)+1j*phi)/d
    return field


def binary_hologram(cfg):
    x,y=grid(cfg); verts,points,phases=tetrahedron(cfg['target'])
    wave=object_wave(x,y,points,phases,cfg['optics']['wavelength_nm'])
    ru,rm=reflections(cfg['media'],cfg['optics']['wavelength_nm'])
    ax,ay=np.sin(np.deg2rad(cfg['optics']['illumination_angles_deg']))
    ref=np.exp(2j*np.pi/(cfg['optics']['wavelength_nm']*1e-6)*(x[None,:]*ax+y[:,None]*ay))
    score=np.real(wave*np.conj(ref)*np.exp(-1j*np.angle(rm-ru)))
    mask=score>np.quantile(score,1-cfg['optimization']['on_fraction'])
    if cfg['optimization']['invert']:
        mask=~mask
    return x,y,mask,verts,points,phases


def sample_grid(values,x,y,xx,yy,order=0):
    ix=(xx-x[0])/(x[1]-x[0]); iy=(yy-y[0])/(y[1]-y[0])
    return map_coordinates(values.astype(float),[iy,ix],order=order,mode='nearest')


def illumination(cfg,x,y):
    ax,ay=np.sin(np.deg2rad(cfg['illumination_angles_deg']))
    inc=np.exp(2j*np.pi/(cfg['wavelength_nm']*1e-6)*(x[None,:]*ax+y[:,None]*ay))
    if cfg['beam_profile']=='gaussian':
        wx,wy=cfg['beam_radius_mm']; cx,cy=cfg['beam_center_mm']
        inc*=np.exp(-((x[None,:]-cx)/wx)**2-((y[:,None]-cy)/wy)**2)
    elif cfg['beam_profile']!='uniform':
        raise ValueError('beam_profile must be gaussian or uniform')
    return inc


def fresnel_roi(field,x,y,wavelength_nm,z,xout,yout):
    """Separable Fresnel integral evaluated directly on a requested output ROI.

    No FFT periodic boundary or implicit output scaling. Returns physical field
    up to the global exp(ikz), with quadrature area and 1/(lambda*z).
    """
    wl=wavelength_nm*1e-6
    a=np.exp(1j*np.pi/(wl*z)*(yout[:,None]-y[None,:])**2)
    b=np.exp(1j*np.pi/(wl*z)*(xout[:,None]-x[None,:])**2)
    return (a@field@b.T)*(x[1]-x[0])*(y[1]-y[0])/(1j*wl*z)
