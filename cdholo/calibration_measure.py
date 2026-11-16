"""Click DVD calibration features in a nearly perpendicular photograph.

The rim ellipse supplies an affine normalization, not a projective camera
calibration. The inner hole independently checks that approximation and supplies
the initial centre. Raw clicks are saved so scale or annotations can be revised.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import least_squares


CSV_FIELDS = ['feature_id', 'kind', 'click_index', 'x_mm', 'y_mm',
              'radius_mm', 'theta_deg', 'angle_unwrapped_deg', 'sigma_mm']


def fit_ellipse(points):
    """Fit an ellipse to at least eight distributed boundary points in pixels."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 8 or not np.isfinite(p).all():
        raise ValueError('Ellipse fitting needs at least eight finite x,y points around the full boundary')
    centre = p.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(np.cov(p-centre, rowvar=False))
    if eigenvalues[0] <= 0 or eigenvalues[0] / eigenvalues[1] < 0.02:
        raise ValueError('Boundary points are nearly collinear; click around the complete circle')
    axes = np.sqrt(2*eigenvalues[::-1])
    angle = np.arctan2(vectors[1, 1], vectors[0, 1])
    initial = np.r_[centre, np.log(axes), angle]

    def residual(params):
        cx, cy, la, lb, phi = params
        a, b = np.exp([la, lb]); c, s = np.cos(phi), np.sin(phi)
        dp = p - [cx, cy]
        u = dp[:, 0]*c + dp[:, 1]*s
        v = -dp[:, 0]*s + dp[:, 1]*c
        rho = np.hypot(u/a, v/b)
        gradient = np.hypot(u/a**2, v/b**2)/np.maximum(rho, 1e-12)
        return (rho-1)/np.maximum(gradient, 1e-12)

    extent = np.ptp(p, axis=0)
    lower = np.r_[p.min(axis=0)-extent, np.log(axes/4), angle-2*np.pi]
    upper = np.r_[p.max(axis=0)+extent, np.log(axes*4), angle+2*np.pi]
    result = least_squares(residual, initial, bounds=(lower, upper), loss='soft_l1', f_scale=1.)
    if not result.success:
        raise ValueError('Ellipse fit did not converge: '+result.message)
    cx, cy, la, lb, phi = result.x
    c, s = np.cos(phi), np.sin(phi)
    rotation = np.array([[c, -s], [s, c]])
    axes = np.exp([la, lb])
    return {'centre_px': np.array([cx, cy]), 'axes_px': axes,
            'rotation': rotation, 'rms_px': float(np.sqrt(np.mean(residual(result.x)**2)))}


def make_transform(rim_points, hole_points, diameter_mm=120., hole_diameter_mm=15., click_sigma_px=2.):
    """Return pixel-to-mm transform plus diagnostics; physical y points upward."""
    if diameter_mm <= 0 or hole_diameter_mm <= 0 or click_sigma_px <= 0:
        raise ValueError('Diameters and click uncertainty must be positive')
    rim = fit_ellipse(rim_points); hole = fit_ellipse(hole_points)
    linear = (diameter_mm/2)*(rim['rotation'] @ np.diag(1/rim['axes_px']) @ rim['rotation'].T)
    # Preserve a right-handed planar display convention despite image y increasing downwards.
    linear = np.diag([1., -1.]) @ linear
    origin = hole['centre_px']
    mm_per_px = float(np.sqrt(abs(np.linalg.det(linear))))
    hp = (np.asarray(hole_points)-origin) @ linear.T
    hr = np.linalg.norm(hp, axis=1)
    centre_offset = float(np.linalg.norm(linear @ (hole['centre_px']-rim['centre_px'])))
    transformed_hole_shape = linear @ hole['rotation'] @ np.diag(hole['axes_px'])
    hole_axes = np.linalg.svd(transformed_hole_shape, compute_uv=False)
    warnings = []
    rim_ratio = float(min(rim['axes_px'])/max(rim['axes_px']))
    hole_ratio = float(min(hole_axes)/max(hole_axes))
    if rim_ratio < .85:
        warnings.append('Strong ellipse: retake closer to perpendicular; affine normalization does not remove perspective')
    if centre_offset > max(.20, 3*mm_per_px):
        warnings.append('Rim and hole centres disagree: check perspective, lens distortion, rim clicks and disc eccentricity')
    if hole_ratio < .97:
        warnings.append('Inner hole remains elliptical after normalization; affine camera approximation may be insufficient')
    if abs(2*np.median(hr)-hole_diameter_mm) > max(.30, 4*mm_per_px):
        warnings.append('Inner-hole diameter disagrees with expected size: check scale and which boundary was clicked')
    if rim['rms_px'] > 2*click_sigma_px:
        warnings.append('Large rim residual: inspect rim points and lens distortion')
    qa = {
        'method': 'outer-rim affine ellipse normalization, origin at independently fitted inner-hole centre',
        'perspective_rectification': False,
        'image_mirroring_verified': False,
        'orientation_note': 'The ellipse cannot detect a mirrored image. Disable camera mirroring and follow the same disc-face/view convention as the target.',
        'diameter_mm': float(diameter_mm), 'expected_hole_diameter_mm': float(hole_diameter_mm),
        'mm_per_pixel_area_equivalent': mm_per_px,
        'coordinate_sigma_mm': float(click_sigma_px*np.linalg.svd(linear, compute_uv=False).max()),
        'click_sigma_px': float(click_sigma_px),
        'rim_fit_rms_px': rim['rms_px'], 'hole_fit_rms_px': hole['rms_px'],
        'rim_minor_major_ratio': rim_ratio, 'rectified_hole_minor_major_ratio': hole_ratio,
        'rim_hole_centre_offset_mm': centre_offset,
        'measured_hole_diameter_mm': float(2*np.median(hr)),
        'warning_messages': warnings,
        'precision_note': 'Pixel scale and click scatter are not a proof of micrometre registration. Diameter error, perspective and distortion are systematic errors.',
        'transform': {'origin_px': origin.tolist(), 'pixel_to_mm': linear.tolist()},
    }
    return {'origin_px': origin, 'pixel_to_mm': linear}, qa, rim, hole


def transform_points(points, transform):
    return (np.asarray(points, dtype=float)-np.asarray(transform['origin_px'])) @ np.asarray(transform['pixel_to_mm']).T


def observation_rows(target, clicks, transform, sigma_mm):
    """Convert clicked ring edges and ordered spoke traces into fitter CSV rows."""
    rows = []; warnings = []; feature_reports = {}
    for feature in target['features']:
        fid = feature['id']; raw = clicks.get(fid, [])
        if not len(raw):
            continue
        p = transform_points(raw, transform)
        r = np.linalg.norm(p, axis=1)
        theta = np.rad2deg(np.arctan2(p[:, 1], p[:, 0]))
        kind = feature['kind']
        if kind == 'ring':
            if len(p) < 6:
                raise ValueError(f'{fid}: use at least six outer-edge points spread around the ring')
            angles = np.sort(np.mod(theta, 360))
            largest_gap = float(np.max(np.diff(np.r_[angles, angles[0]+360])))
            if largest_gap > 180:
                warnings.append(f'{fid}: ring clicks cover less than half the circle; add points at more azimuths')
            unwrapped = None
            feature_reports[fid] = {'kind': kind, 'clicks': len(p), 'median_radius_mm': float(np.median(r)),
                                    'radius_scatter_mm': float(np.std(r)), 'largest_unsampled_angle_deg': largest_gap}
        elif kind == 'spoke':
            if len(p) < 8:
                raise ValueError(f'{fid}: trace at least eight points from inner to outer radius')
            if np.any(np.diff(r) <= 0):
                raise ValueError(f'{fid}: spoke radius must increase with every click; trace inner to outer, or undo the misplaced point')
            unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta)))
            jump = float(np.max(np.abs(np.diff(unwrapped))))
            if jump > 90:
                raise ValueError(f'{fid}: an angular step exceeds 90 degrees; add points along the same continuous stripe')
            feature_reports[fid] = {'kind': kind, 'clicks': len(p), 'max_observed_angle_step_deg': jump,
                                    'radius_span_mm': [float(r.min()), float(r.max())],
                                    'unwrap_assumption': 'No missed full turns between consecutive clicks; cannot be established from endpoints alone'}
        else:
            continue
        for index, ((xx, yy), radius, angle) in enumerate(zip(p, r, theta)):
            rows.append({'feature_id': fid, 'kind': kind, 'click_index': index,
                         'x_mm': float(xx), 'y_mm': float(yy), 'radius_mm': float(radius),
                         'theta_deg': float(angle), 'angle_unwrapped_deg': '' if unwrapped is None else float(unwrapped[index]),
                         'sigma_mm': float(sigma_mm)})
    if not rows:
        raise ValueError('No ring or spoke measurements were supplied')
    return rows, warnings, feature_reports


def _ellipse_points(ellipse):
    t = np.linspace(0, 2*np.pi, 361)
    return ellipse['centre_px'] + (np.column_stack((np.cos(t), np.sin(t)))*ellipse['axes_px']) @ ellipse['rotation'].T


def _collect_points(fig, ax, prompt, initial=None):
    """Toolbar zoom/pan stays available while collecting a click sequence."""
    points = list(initial or []); cancelled = []
    ax.set_title(prompt+'\nLeft: add | right/backspace: undo | Enter: finish/skip | toolbar: zoom/pan', fontsize=10)
    marks, = ax.plot([], [], 'o-', color='#ff684d', markersize=4, linewidth=.7)

    def redraw():
        p = np.array(points)
        marks.set_data(p[:, 0] if len(p) else [], p[:, 1] if len(p) else [])
        fig.canvas.draw_idle()

    def click(event):
        toolbar = getattr(fig.canvas.manager, 'toolbar', None)
        if event.inaxes is not ax or (toolbar and toolbar.mode): return
        if event.button == 1: points.append([float(event.xdata), float(event.ydata)])
        elif event.button == 3 and points: points.pop()
        redraw()

    def key(event):
        if event.key in ('enter', 'return'): fig.canvas.stop_event_loop()
        elif event.key in ('backspace', 'delete') and points:
            points.pop(); redraw()
        elif event.key == 'escape':
            cancelled.append(True); fig.canvas.stop_event_loop()

    def closed(event):
        cancelled.append(True); fig.canvas.stop_event_loop()

    ids = [fig.canvas.mpl_connect('button_press_event', click), fig.canvas.mpl_connect('key_press_event', key),
           fig.canvas.mpl_connect('close_event', closed)]
    redraw(); fig.canvas.start_event_loop(timeout=0)
    for cid in ids: fig.canvas.mpl_disconnect(cid)
    marks.remove()
    if cancelled: raise RuntimeError('Measurement cancelled; no observations CSV was written')
    return points


def _preview(image, target, session, rim, hole, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    for ellipse, label, color in [(rim, 'outer rim fit', '#23de83'), (hole, 'inner hole fit', '#21c9ee')]:
        e = _ellipse_points(ellipse); ax.plot(e[:, 0], e[:, 1], color=color, linewidth=1, label=label)
    for feature_index, feature in enumerate(target['features']):
        points = np.asarray(session['features'].get(feature['id'], []))
        if not len(points): continue
        line, = ax.plot(points[:, 0], points[:, 1], 'o' if feature['kind']=='ring' else 'o-', markersize=3, linewidth=.8)
        anchor = points[(feature_index*3) % len(points)] if feature['kind']=='ring' else points[0]
        ax.annotate(feature['id'], anchor, color='white', fontsize=9,
                    bbox={'facecolor': line.get_color(), 'alpha': .8, 'pad': 2, 'edgecolor': 'none'})
    ax.set(xlim=(0, image.shape[1]), ylim=(image.shape[0], 0), title='Calibration clicks - affine photo normalization only')
    ax.legend(loc='upper right'); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def measure_photo(image_path, target_path, output_path, diameter_mm=120., *, features=None,
                  clicks_path=None, hole_diameter_mm=15., click_sigma_px=2.):
    """Interactively measure, or replay an existing .clicks.json without a GUI."""
    image_path = Path(image_path); target_path = Path(target_path); output_path = Path(output_path)
    target = json.loads(target_path.read_text(encoding='utf-8'))
    if target.get('schema') != 'dvd-calibration-v1': raise ValueError('Expected dvd-calibration-v1 target JSON')
    with Image.open(image_path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
    selected = set(features.split(',')) if isinstance(features, str) else set(features or [f['id'] for f in target['features']])
    known = {f['id'] for f in target['features']}
    if selected-known: raise ValueError('Unknown feature IDs: '+', '.join(sorted(selected-known)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if clicks_path:
        session = json.loads(Path(clicks_path).read_text(encoding='utf-8'))
        session['features'] = {k: v for k, v in session['features'].items() if k in selected}
    else:
        import matplotlib.pyplot as plt
        if plt.get_backend().lower() in ('agg', 'module://matplotlib_inline.backend_inline'):
            raise RuntimeError('Interactive photo measurement needs a desktop matplotlib backend (for example TkAgg); use --clicks-json for replay')
        fig, ax = plt.subplots(figsize=(11, 9)); ax.imshow(image); ax.set_axis_off()
        plt.show(block=False)
        session = {'schema': 'dvd-calibration-clicks-v1', 'image': str(image_path.resolve()),
                   'target': str(target_path.resolve()), 'features': {}}
        try:
            for key, instruction in [('outer_rim_px', 'OUTER DISC EDGE: >=8 points all around the rim; avoid shadows'),
                                     ('inner_hole_px', 'INNER 15 mm HOLE EDGE: >=8 points all around; not the transparent hub')]:
                retry_points = None
                while True:
                    points = _collect_points(fig, ax, instruction, initial=retry_points)
                    try:
                        fit_ellipse(points)
                    except ValueError as error:
                        print(str(error), flush=True)
                        retry_points = points
                        continue
                    session[key] = points
                    break
            transform, qa, rim, hole = make_transform(session['outer_rim_px'], session['inner_hole_px'], diameter_mm, hole_diameter_mm, click_sigma_px)
            for warning in qa['warning_messages']: print('Photo QA: '+warning, flush=True)
            for feature in target['features']:
                fid = feature['id']
                if fid not in selected: continue
                if feature['kind']=='ring':
                    prompt = f"{fid}: OUTER edge of annulus nominal r={feature['nominal_radius_mm']:.3f} mm; >=6 distributed points, avoid spokes"
                elif feature['kind']=='spoke':
                    prompt = f"{fid}: one continuous stripe, INNER to OUTER, >=8 points; angular step <90 deg, no skipped turns"
                else: continue
                retry_points = None
                while True:
                    points = _collect_points(fig, ax, prompt, initial=retry_points)
                    if not points: break
                    try:
                        observation_rows({'features': [feature]}, {fid: points}, transform, qa['coordinate_sigma_mm'])
                    except ValueError as error:
                        print(str(error), flush=True); prompt = f'{error}. Please reselect {fid}.'
                        retry_points = points
                        continue
                    session['features'][fid] = points
                    break
        finally:
            plt.close(fig)
    transform, qa, rim, hole = make_transform(session['outer_rim_px'], session['inner_hole_px'], diameter_mm, hole_diameter_mm, click_sigma_px)
    rows, warnings, feature_reports = observation_rows(target, session['features'], transform, qa['coordinate_sigma_mm'])
    qa['warning_messages'].extend(warnings); qa['features'] = feature_reports
    qa['scale_note'] = '120 mm is nominal unless independently measured; supply --diameter-mm from a physical measurement'
    qa['image_path'] = str(image_path.resolve()); qa['target_path'] = str(target_path.resolve())
    qa['image_exif_orientation_applied'] = True
    qa['observations_count'] = len(rows)
    session['measurement'] = {'diameter_mm': diameter_mm, 'hole_diameter_mm': hole_diameter_mm, 'click_sigma_px': click_sigma_px}
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS); writer.writeheader(); writer.writerows(rows)
    output_path.with_suffix('.clicks.json').write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding='utf-8')
    output_path.with_suffix('.qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    _preview(image, target, session, rim, hole, output_path.with_suffix('.preview.png'))
    print(f'Saved {len(rows)} observations: {output_path}', flush=True)
    for warning in qa['warning_messages']: print('Photo QA: '+warning, flush=True)
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True); parser.add_argument('--target', required=True)
    parser.add_argument('--output', default='output/calibration/observations.csv')
    parser.add_argument('--diameter-mm', type=float, default=120.)
    parser.add_argument('--hole-diameter-mm', type=float, default=15.)
    parser.add_argument('--click-sigma-px', type=float, default=2.)
    parser.add_argument('--features', help='Comma-separated feature IDs, e.g. R01,R05,R09,S01')
    parser.add_argument('--clicks-json', help='Replay saved raw clicks without opening the interactive window')
    args = parser.parse_args()
    measure_photo(args.image, args.target, args.output, args.diameter_mm, features=args.features,
                  clicks_path=args.clicks_json, hole_diameter_mm=args.hole_diameter_mm, click_sigma_px=args.click_sigma_px)


if __name__ == '__main__': main()
