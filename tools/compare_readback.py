"""Compare a securely ripped PCM WAV with generated WAV, accounting for sample offset."""
import argparse
import json
import wave
import numpy as np


def read(path):
    with wave.open(path,'rb') as f:
        if (f.getnchannels(),f.getsampwidth(),f.getframerate())!=(2,2,44100):
            raise ValueError('Both files must be 44.1 kHz stereo PCM16 WAV')
        return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').reshape(-1,2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('original'); p.add_argument('readback'); p.add_argument('--max-offset',type=int,default=3000)
    a=p.parse_args(); original=read(a.original); got=read(a.readback)
    # Search near non-constant material, rather than the silent/repetitive inner fill.
    variation=np.flatnonzero(np.any(original[1:]!=original[:-1],axis=1))
    if not len(variation): raise ValueError('Reference contains no varying PCM for offset detection')
    start=max(a.max_offset,int(variation[len(variation)//2]))
    start=min(start,min(len(original),len(got))-a.max_offset-2048)
    if start<a.max_offset: raise ValueError('Files too short for this offset search')
    offsets=range(-a.max_offset,a.max_offset+1)
    best=min(offsets,key=lambda off:np.count_nonzero(original[start:start+2048]!=got[start+off:start+off+2048]))
    lo=max(0,-best); hi=min(len(original),len(got)-best)
    errors=int(np.count_nonzero(original[lo:hi]!=got[lo+best:hi+best]))
    print(json.dumps({'readback_offset_stereo_samples':best,'compared_stereo_samples':hi-lo,
        'different_int16_values':errors,'exact_overlap_match':errors==0,
        'excluded_reference_samples_at_edges':len(original)-(hi-lo),
        'meaning':'PCM integrity only; this does not measure physical EFM merges or optical marks.'},indent=2))
    raise SystemExit(0 if errors==0 else 1)


if __name__=='__main__': main()
