"""CD-DA bytes -> CIRC -> EFM -> merging bits -> NRZI recording marks.

The reference is ECMA-130, 2nd edition, Annexes C/D/E. Annex C explicitly
uses the same CIRC as IEC 908. F1 input is high-byte-first 16-bit stereo PCM;
WAV little-endian byte pairs must therefore be swapped before calling here.
There is NO CD-ROM sector scrambler in this CD-DA path.

This is a deterministic reference encoder, not a capture of a particular drive:
the drive's DSV lookahead, subcode alignment, write offset and initial NRZI sign
can differ. Only the systematic-data interleaving and EFM words are invariant.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

CHANNEL_BITS_PER_FRAME = 588
FRAMES_PER_SECOND = 7350
CHANNEL_BIT_RATE = 4321800
# Longest *data-byte* route is 108. An even C1 parity byte is delayed once
# more and can depend on that 108-frame-old input: 109 history frames needed.
MAX_CIRC_DELAY = 109
SYNC = tuple(map(int, "100000000001000000000010"))
SUBCODE_SYNC = (tuple(map(int, "00100000000001")),
                tuple(map(int, "00000000010010")))
_TABLE = json.loads(Path(__file__).with_name("efm_table.json").read_text())
EFM = np.asarray([list(map(int, word)) for word in _TABLE["words"]], dtype=np.uint8)
EFM_TRANSITIONS = EFM.sum(axis=1)


@lru_cache(maxsize=1)
def _mean_merge_table():
    """Pair-local RLL allowed-merge mean, NOT the drive's DSV decision.

    Used only by the homogenized envelope surrogate. Full encode_channel below
    also checks false sync and the preceding 23 channel bits.
    """
    first=np.argmax(EFM,axis=1)
    last=13-np.argmax(EFM[:,::-1],axis=1)
    total=np.zeros((256,256)); count=np.zeros((256,256))
    gap=17+first[None,:]-last[:,None]
    valid=(gap>=3)&(gap<=11); count+=valid
    for pos in (14,15,16):
        a=pos-last[:,None]; b=17+first[None,:]-pos
        valid=(a>=3)&(a<=11)&(b>=3)&(b<=11)
        total+=valid; count+=valid
    if np.any(count==0): raise AssertionError('No legal pair-local EFM merge')
    return total/count


def mean_merge_transitions(left,right):
    return _mean_merge_table()[left,right]

# Output order before the Q slots are inserted; C.1 and C.4 disambiguate the
# typographical mistakes in the printed V_Q expression of Figure C.3.
_PERM = np.array([0, 1, 8, 9, 16, 17, 2, 3, 10, 11, 18, 19,
                  4, 5, 12, 13, 20, 21, 6, 7, 14, 15, 22, 23])
_DATA_SLOTS = np.r_[0:12, 16:28]
DATA_POSITIONS = np.empty(24, dtype=np.int64)
DATA_DELAYS = np.empty(24, dtype=np.int64)
for _rank, (_source, _slot) in enumerate(zip(_PERM, _DATA_SLOTS)):
    DATA_POSITIONS[_source] = _slot
    DATA_DELAYS[_source] = 4 * _slot + (2 if _rank < 12 else 0) + (1 if _slot % 2 == 0 else 0)
DATA_CHANNEL_OFFSETS = 44 + 17 * DATA_POSITIONS


def _make_gf():
    exp = np.zeros(510, dtype=np.uint8)
    log = np.zeros(256, dtype=np.int16)
    value = 1
    for i in range(255):
        exp[i] = value
        log[value] = i
        value <<= 1
        if value & 256:
            value ^= 0x11D
    exp[255:] = exp[:255]
    mul = exp[log[:, None] + log[None, :]]
    mul[0, :] = 0
    mul[:, 0] = 0
    return exp, log, mul


_EXP, _LOG, _MUL = _make_gf()


def parity_check_matrix(length):
    """Four RS checks, roots alpha**0 ... alpha**3 (C.7)."""
    return _EXP[(np.arange(4)[:, None] * np.arange(length - 1, -1, -1)) % 255]


def _gf_inverse(matrix):
    n = len(matrix)
    aug = np.concatenate([matrix.copy(), np.eye(n, dtype=np.uint8)], axis=1)
    for col in range(n):
        pivot = col + np.flatnonzero(aug[col:, col])[0]
        aug[[col, pivot]] = aug[[pivot, col]]
        inv = _EXP[255 - int(_LOG[aug[col, col]])]
        aug[col] = _MUL[inv, aug[col]]
        for row in range(n):
            if row != col:
                aug[row] ^= _MUL[aug[row, col], aug[col]]
    return aug[:, n:]


def _parity_coefficients(length, slots):
    check = parity_check_matrix(length)
    data = np.array([i for i in range(length) if i not in slots])
    inverse = _gf_inverse(check[:, slots])
    coeff = np.zeros((4, len(data)), dtype=np.uint8)
    for k in range(4):
        coeff ^= _MUL[inverse[:, k, None], check[k, data][None, :]]
    return data, coeff


_C2_DATA, _C2_COEFF = _parity_coefficients(28, [12, 13, 14, 15])
_C1_DATA, _C1_COEFF = _parity_coefficients(32, [28, 29, 30, 31])


def _parity(data, coeff):
    out = np.zeros((len(data), 4), dtype=np.uint8)
    for j in range(data.shape[1]):
        out ^= _MUL[coeff[:, j][None, :], data[:, j, None]]
    return out


def rs_syndromes(words):
    """Independent matrix check for an uninverted C1/C2 codeword batch."""
    words = np.asarray(words, dtype=np.uint8)
    h = parity_check_matrix(words.shape[1])
    return _parity(words, h)


def _delay_columns(data, delays):
    result = np.zeros_like(data)
    for j, delay in enumerate(delays):
        if delay == 0:
            result[:, j] = data[:, j]
        elif delay < len(data):
            result[delay:, j] = data[:-delay, j]
    return result


def encode_circ(f1, *, history=None):
    """Return N x 32 F2 bytes from N x 24 canonical F1 bytes.

    Without history, prehistory is zero PCM and delay registers start at zero.
    Supply the last 109 F1 frames as history for continuous block processing.
    No tail is appended: append 109 F1 frames externally to flush a track tail.
    Memory scales with the block size; blocks of 16384 frames are convenient.
    """
    f1 = np.asarray(f1, dtype=np.uint8)
    if f1.ndim != 2 or f1.shape[1] != 24:
        raise ValueError("CIRC input must have shape (N, 24)")
    prefix = 0
    if history is not None:
        history = np.asarray(history, dtype=np.uint8)
        if history.ndim != 2 or history.shape[1] != 24:
            raise ValueError("CIRC history must have shape (M, 24)")
        history = history[-MAX_CIRC_DELAY:]
        prefix = len(history)
        f1 = np.concatenate([history, f1])
    ordered = _delay_columns(f1[:, _PERM], [2] * 12 + [0] * 12)
    c2 = np.empty((len(f1), 28), dtype=np.uint8)
    c2[:, _C2_DATA] = ordered
    c2[:, 12:16] = _parity(ordered, _C2_COEFF)
    delayed = _delay_columns(c2, np.arange(28) * 4)
    c1 = np.empty((len(f1), 32), dtype=np.uint8)
    c1[:, :28] = delayed
    c1[:, 28:] = _parity(delayed, _C1_COEFF)
    out = _delay_columns(c1, (np.arange(32) % 2 == 0).astype(int))
    out[:, 12:16] ^= 255
    out[:, 28:32] ^= 255
    return out[prefix:]


def precompensate_systematic(desired_f2, *, fill=0):
    """Invert just the 24 systematic byte routes; parity is computed later.

    desired_f2 is (N,32); parity slots are ignored. The last 108 input frames
    retain fill where they cannot affect an output inside this N-frame window.
    """
    desired_f2 = np.asarray(desired_f2, dtype=np.uint8)
    if desired_f2.ndim != 2 or desired_f2.shape[1] != 32:
        raise ValueError("desired_f2 must have shape (N, 32)")
    out = np.full((len(desired_f2), 24), fill, dtype=np.uint8)
    for source, (slot, delay) in enumerate(zip(DATA_POSITIONS, DATA_DELAYS)):
        if delay < len(out):
            out[:-delay, source] = desired_f2[delay:, slot]
    return out


def pcm_le_to_f1(pcm):
    """WAV payload bytes to high-byte-first CIRC frames, no sample processing."""
    data = np.frombuffer(pcm, dtype=np.uint8)
    if len(data) % 24:
        raise ValueError("PCM payload must contain a whole number of 24-byte F1 frames")
    return data.reshape(-1, 2)[:, ::-1].copy().reshape(-1, 24)


def f1_to_pcm_le(f1):
    return np.asarray(f1, dtype=np.uint8).reshape(-1, 2)[:, ::-1].tobytes()


def _bcd(value):
    return (value // 10 << 4) | (value % 10)


def _msf(sector):
    return [_bcd(sector // 4500), _bcd(sector // 75 % 60), _bcd(sector % 75)]


@lru_cache(maxsize=1024)
def subcode_q(sector, track=1, index=1, absolute_offset=150, control=0):
    """Nominal Q, one audio track; P/R-W are zero in the program area.

    Actual recorder subcode timing is an independent alignment uncertainty.
    This function does not model lead-in, pregap, lead-out or CD-TEXT.
    """
    if sector < 0 or sector + absolute_offset >= 100 * 4500:
        raise ValueError("Subcode time is outside two-digit BCD MSF range")
    q = [(control << 4) | 1, _bcd(track), _bcd(index), *_msf(sector),
         0, *_msf(sector + absolute_offset)]
    crc = 0
    for byte in q:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xFFFF
    crc ^= 0xFFFF
    return tuple(q + [crc >> 8, crc & 255])


def _subcode_word(frame, absolute_offset):
    block, pos = divmod(frame, 98)
    if pos < 2:
        return SUBCODE_SYNC[pos]
    bit = pos - 2
    q = subcode_q(block, absolute_offset=absolute_offset)
    return tuple(EFM[((q[bit // 8] >> (7 - bit % 8)) & 1) << 6])


def nrzi(channel, initial_mark=0):
    """Channel 1 means a transition; result 1 means a recording mark."""
    channel = np.asarray(channel, dtype=np.uint8)
    return (np.cumsum(channel, dtype=np.uint64) % 2).astype(np.uint8) ^ initial_mark


def _valid_join(tail, addition, allow_final_sync=False):
    joined = tail + addition
    ones = [i for i, bit in enumerate(joined) if bit]
    if any(not 3 <= b - a <= 11 for a, b in zip(ones, ones[1:])):
        return False
    for i in range(max(0, len(tail) - 23), len(joined) - 23):
        if joined[i:i + 24] == SYNC:
            if not (allow_final_sync and i == len(joined) - 24):
                return False
    return True


@lru_cache(maxsize=65536)
def _merge_choices(tail, word, final_sync):
    choices = []
    candidates = (0, 4) if final_sync else (0, 1, 2, 4)
    for merge in candidates:
        prefix = tuple((merge >> shift) & 1 for shift in (2, 1, 0))
        full = prefix + word
        if _valid_join(tail, full, final_sync):
            sent = prefix if final_sync else full
            sign = 1
            total = 0
            for bit in sent:
                if bit:
                    sign = -sign
                total += sign
            choices.append((sent, sign, total, merge))
    return choices


def encode_channel(f2, *, start_frame=0, absolute_offset=150,
                   initial_mark=0, initial_dsv=0, return_state=False):
    """Encode an F2 excerpt to flat uint8 transition bits.

    Uses the Annex E greedy minimum-|DSV| strategy. Ties prefer a transition,
    then numeric merge order. A drive can use a different legal strategy.
    Excerpts start at a frame sync; retain NRZI/DSV via returned state if needed.
    Preceding frame's trailing merge must be generated with this encoder for
    an exact stream continuation. The 588th bit closes the last merge group.
    """
    f2 = np.asarray(f2, dtype=np.uint8)
    if f2.ndim != 2 or f2.shape[1] != 32:
        raise ValueError("F2 must have shape (N, 32)")
    out = np.empty(len(f2) * 588, dtype=np.uint8)
    sign = -1 if initial_mark else 1
    dsv = int(initial_dsv)
    cursor = 0
    tail = ()

    def emit(bits):
        nonlocal sign, dsv, cursor, tail
        out[cursor:cursor + len(bits)] = bits
        cursor += len(bits)
        for bit in bits:
            if bit:
                sign = -sign
            dsv += sign
        tail = (tail + bits)[-23:]

    for n, row in enumerate(f2):
        emit(SYNC)
        words = [_subcode_word(start_frame + n, absolute_offset)]
        words.extend(tuple(EFM[b]) for b in row)
        for word in words:
            choices = _merge_choices(tail, word, False)
            if not choices:
                raise RuntimeError("No legal EFM merge; inconsistent preceding state")
            best = min(choices, key=lambda c: (abs(dsv + sign * c[2]), c[3] == 0, c[3]))
            emit(best[0])
        choices = _merge_choices(tail, SYNC, True)
        if not choices:
            raise RuntimeError("No legal merging bits before frame sync")
        best = min(choices, key=lambda c: (abs(dsv + sign * c[2]), c[3] == 0, c[3]))
        emit(best[0])
    state = {"initial_mark": int(sign < 0), "initial_dsv": dsv,
             "start_frame": start_frame + len(f2)}
    return (out, state) if return_state else out


def mark_runs(channel, initial_mark=0):
    """Return (start_channel_bit, length_channel_bits, is_mark) run rows."""
    marks = nrzi(channel, initial_mark)
    if not len(marks):
        return np.empty((0, 3), dtype=np.int64)
    edges = np.r_[0, np.flatnonzero(np.diff(marks)) + 1, len(marks)]
    return np.column_stack([edges[:-1], np.diff(edges), marks[edges[:-1]]])
