"""DVD-R EFMPlus reference: ECMA-359 §§20-22, Annex G.

Full samples use the standard two-stream DSV procedure (ties choose stream 1).
The actual writer can have different tie decisions, initial phase and physical
write strategy. State averages are only a coarse optical-envelope surrogate.
The tables were extracted independently from ECMA-267 Annex G, shared by DVD-R.
"""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

_TABLE = json.loads(Path(__file__).with_name('efmplus_table.json').read_text())
MAIN = np.array([[int(w,2) for w,s in row] for row in _TABLE['main']],dtype=np.uint16)
MAIN_NEXT = np.array([[s for w,s in row] for row in _TABLE['main']],dtype=np.uint8)
SUB = np.array([[int(w,2) for w,s in row] for row in _TABLE['substitution']],dtype=np.uint16)
SUB_NEXT = np.array([[s for w,s in row] for row in _TABLE['substitution']],dtype=np.uint8)
MAIN_TRANSITIONS = np.array([[int(w).bit_count() for w in row] for row in MAIN],float)
SUB_TRANSITIONS = np.array([[int(w).bit_count() for w in row] for row in SUB],float)
EFMPLUS_TRANSITIONS = MAIN_TRANSITIONS.mean(axis=1)
EFMPLUS_TRANSITIONS[:88] = (MAIN_TRANSITIONS[:88].mean(axis=1)+SUB_TRANSITIONS.mean(axis=1))/2
EFMPLUS_MIN_TRANSITIONS = MAIN_TRANSITIONS.min(axis=1)
EFMPLUS_MAX_TRANSITIONS = MAIN_TRANSITIONS.max(axis=1)
EFMPLUS_MIN_TRANSITIONS[:88] = np.minimum(EFMPLUS_MIN_TRANSITIONS[:88],SUB_TRANSITIONS.min(axis=1))
EFMPLUS_MAX_TRANSITIONS[:88] = np.maximum(EFMPLUS_MAX_TRANSITIONS[:88],SUB_TRANSITIONS.max(axis=1))

SYNC_SEQUENCE = np.array([0,5,1,5,2,5,3,5,4,5,1,6,2,6,3,6,4,6,1,7,2,7,3,7,4,7],np.uint8)
BITS_PER_SYNC_FRAME = 1488
BITS_PER_SECTOR = 38688
BYTES_PER_RECORDING_FRAME = 2366
# [state group:1/2 or3/4][sync type][primary/secondary]; common low word=0011hex.
# SY6 secondary for states3/4 is 0x80440011, as printed in ECMA-359 Table5.
_SYNC_PREFIX = [
    [(0x1244,0x1204),(0x0404,0x0444),(0x1004,0x1044),(0x0804,0x0844),
     (0x2004,0x2044),(0x2244,0x2204),(0x2484,0x2084),(0x2444,0x2404)],
    [(0x9204,0x9244),(0x8444,0x8404),(0x9044,0x9004),(0x8244,0x8204),
     (0x8844,0x8804),(0x8904,0x8104),(0x9084,0x8044),(0x8884,0x8084)],
]
SYNC = np.array([[[hi<<16|0x11 for hi in pair] for pair in group] for group in _SYNC_PREFIX],dtype=np.uint32)


@lru_cache(maxsize=4096)
def _word_properties(word,length=16):
    bits=tuple((word>>i)&1 for i in range(length-1,-1,-1))
    sign=1; total=0
    for bit in bits:
        if bit: sign=-sign
        total+=sign
    first=bits.index(1); trailing=(word&-word).bit_length()-1
    return bits,total,sign,first,trailing


def _representations(byte,state,trailing):
    """Distinct legal possibilities, with table state numbers kept 1-based."""
    result=[(int(MAIN[byte,state-1]),int(MAIN_NEXT[byte,state-1]))]
    if byte<88:
        result.append((int(SUB[byte,state-1]),int(SUB_NEXT[byte,state-1])))
    elif state in (1,4):
        other=4 if state==1 else 1
        result.append((int(MAIN[byte,other-1]),int(MAIN_NEXT[byte,other-1])))
    return list(dict.fromkeys((word,nxt) for word,nxt in result
        if 3<=trailing+_word_properties(word)[3]+1<=11))


def transitions(state,byte,substitution=False):
    """Return the standard (16-bit word,next state) for a specified table entry."""
    if state not in (1,2,3,4) or not 0<=byte<256:
        raise ValueError('state must be1..4 and byte must be0..255')
    if substitution:
        if byte>=88: raise ValueError('Substitution table only covers bytes0..87')
        return int(SUB[byte,state-1]),int(SUB_NEXT[byte,state-1])
    return int(MAIN[byte,state-1]),int(MAIN_NEXT[byte,state-1])


class _Stream:
    __slots__=('words','dsv','sign','state','trailing','sync_choice')
    def __init__(self,word,dsv,sign,sync_choice):
        self.words=[word]; self.dsv=dsv; self.sign=sign
        self.state=1; self.trailing=0; self.sync_choice=sync_choice
        self._advance(word,32)

    def _advance(self,word,length):
        _,total,sign,_,trailing=_word_properties(word,length)
        self.dsv+=self.sign*total; self.sign*=sign; self.trailing=trailing

    def copy(self):
        other=object.__new__(_Stream)
        other.words=self.words.copy(); other.dsv=self.dsv; other.sign=self.sign
        other.state=self.state; other.trailing=self.trailing; other.sync_choice=self.sync_choice
        return other

    def append(self,word,next_state):
        self.words.append(word); self._advance(word,16); self.state=next_state


def _encode_sync_frame(data,sync_type,state,sign,dsv):
    group=0 if state in (1,2) else 1
    sync_words=SYNC[group,sync_type]
    streams=[_Stream(int(sync_words[i]),dsv,sign,i) for i in (0,1)]
    for byte in data:
        byte=int(byte)
        candidates=[_representations(byte,s.state,s.trailing) for s in streams]
        if any(not c for c in candidates): raise AssertionError('No legal EFMPlus continuation')
        chosen=0 if abs(streams[0].dsv)<=abs(streams[1].dsv) else 1
        if len(candidates[chosen])==2:
            seed=streams[chosen]
            streams=[seed.copy(),seed.copy()]
            for stream,entry in zip(streams,candidates[chosen]): stream.append(*entry)
        else:
            for stream,entries in zip(streams,candidates): stream.append(*entries[0])
    chosen=min(streams,key=lambda s:abs(s.dsv))
    # §22 final primary/secondary inversion if DSV lies outside [-64,+63].
    if not -64<=chosen.dsv<=63:
        alternate=chosen.words.copy(); alternate[0]=int(sync_words[1-chosen.sync_choice])
        trial_dsv=dsv; trial_sign=sign
        for i,word in enumerate(alternate):
            _,delta,parity,_,_=_word_properties(word,32 if i==0 else 16)
            trial_dsv+=trial_sign*delta; trial_sign*=parity
        if abs(trial_dsv)<abs(chosen.dsv):
            chosen.words=alternate; chosen.dsv=trial_dsv; chosen.sign=trial_sign
    bits=np.concatenate([np.array(_word_properties(word,32 if i==0 else 16)[0],np.uint8)
                         for i,word in enumerate(chosen.words)])
    return bits,chosen.state,chosen.sign,chosen.dsv


def encode_recording_frames(recording,*,initial_state=1,initial_mark=0,initial_dsv=0,return_state=False):
    """Encode N recording frames (N,2366 or N,13,182) into N*38688 NRZ bits.

    Bits equal1 mean transitions; call nrzi() to get mark occupancy. Each sector
    has26 sync frames, 32sync bits+91*16 data bits each. State and DSV persist.
    """
    data=np.asarray(recording,dtype=np.uint8)
    if data.size%2366: raise ValueError('Recording data must contain whole2366-byte frames')
    if initial_state not in (1,2,3,4) or initial_mark not in (0,1):
        raise ValueError('Invalid initial EFMPlus state or NRZI polarity')
    data=data.reshape(-1,26,91); out=np.empty((len(data),26,1488),np.uint8)
    state=initial_state; sign=1 if initial_mark==0 else -1; dsv=int(initial_dsv)
    for sector,rows in enumerate(data):
        for index,row in enumerate(rows):
            bits,state,sign,dsv=_encode_sync_frame(row,int(SYNC_SEQUENCE[index]),state,sign,dsv)
            out[sector,index]=bits
    final={'initial_state':state,'initial_mark':int(sign<0),'initial_dsv':dsv}
    return (out.reshape(-1),final) if return_state else out.reshape(-1)


def nrzi(channel,initial_mark=0):
    return (np.cumsum(np.asarray(channel,np.uint8),dtype=np.uint64)%2).astype(np.uint8)^initial_mark


def mark_runs(channel,initial_mark=0):
    marks=nrzi(channel,initial_mark)
    if not len(marks): return np.empty((0,3),dtype=np.int64)
    edges=np.r_[0,np.flatnonzero(np.diff(marks))+1,len(marks)]
    return np.column_stack([edges[:-1],np.diff(edges),marks[edges[:-1]]])
