import argparse
from .config import load_config

def main():
    parser=argparse.ArgumentParser(description='DVD-R / CD-R tetrahedron hologram and DVD calibration')
    commands=parser.add_subparsers(dest='command',required=True)
    build=commands.add_parser('build',help='Generate a tetrahedron hologram disc image')
    build.add_argument('--config',default='configs/tetrahedron_dvd.json');build.add_argument('--output')
    cal=commands.add_parser('calibration-build',help='Generate known-address rings and spokes')
    cal.add_argument('--config',default='configs/tetrahedron_dvd.json')
    cal.add_argument('--pattern',default='configs/calibration_dvd.json')
    cal.add_argument('--output',default='output/calibration_dvd')
    measure=commands.add_parser('calibration-measure',help='Click calibration measurements on a photograph')
    measure.add_argument('--image',required=True)
    measure.add_argument('--target',default='output/calibration_dvd/target.json')
    measure.add_argument('--output',default='output/calibration_dvd/measurements.csv')
    measure.add_argument('--diameter-mm',type=float,default=120.)
    measure.add_argument('--hole-diameter-mm',type=float,default=15.)
    measure.add_argument('--click-sigma-px',type=float,default=2.)
    measure.add_argument('--features',help='Comma-separated feature IDs, e.g. R01,R05,R09,S01')
    measure.add_argument('--clicks','--clicks-json',dest='clicks',help='Replay a saved click-session JSON instead of opening the GUI')
    fit=commands.add_parser('calibration-fit',help='Fit geometry and export a new configuration')
    fit.add_argument('--target',default='output/calibration_dvd/target.json')
    fit.add_argument('--measurements',required=True)
    fit.add_argument('--mode',choices=['coarse','refine'],default='refine')
    fit.add_argument('--direction',choices=['auto','1','-1'],default='auto',
                     help='Infer chirality from two asymmetric spokes, or specify the actual disc direction')
    fit.add_argument('--output',default='output/calibration_fit')
    args=parser.parse_args()
    if args.command=='calibration-build':
        from .calibration_target import build_calibration
        build_calibration(load_config(args.config),args.pattern,args.output)
    elif args.command=='calibration-measure':
        from .calibration_measure import measure_photo
        measure_photo(args.image,args.target,args.output,diameter_mm=args.diameter_mm,
                      features=args.features.split(',') if args.features else None,clicks_path=args.clicks,
                      hole_diameter_mm=args.hole_diameter_mm,click_sigma_px=args.click_sigma_px)
    elif args.command=='calibration-fit':
        from .calibration_fit import fit_calibration
        report=fit_calibration(args.target,args.measurements,args.output,mode=args.mode,direction=args.direction)
        print(f"Saved {args.output}/calibrated_config.json and fit_report.json")
        print(f"Ring radial RMS: {report['residuals']['ring_radial_rms_um']:.3f} um")
        if report['held_out_validation']:
            print(f"Held-out trace RMS: {report['held_out_validation']['tangential_rms_um']:.3f} um")
        print('Geometry fit only; inspect photo QA and residuals before rebuilding the hologram.')
    else:
        cfg=load_config(args.config)
        if cfg.get('format')=='DVD-R':
            from .dvd_pipeline import run
            default_output='output/tetrahedron_dvd'
        else:
            from .pipeline import run
            default_output='output/tetrahedron'
        run(cfg,args.output or default_output)

if __name__=='__main__': main()
