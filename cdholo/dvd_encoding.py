"""DVD-R single-layer data-zone framing, scrambling and RSPC.

Independent implementation of ECMA-359 (2004), clauses 16--19, 24.2.
These routines predict the firmware encoding of 2048-byte logical sectors;
their ECC bytes are NOT written into a host ISO image.
"""
from functools import lru_cache
import numpy as np

PAYLOAD_BYTES = 2048
DATA_FRAME_BYTES = 2064
RECORDING_FRAME_BYTES = 2366
SECTORS_PER_ECC_BLOCK = 16
CHANNEL_BITS_PER_SECTOR = 38688
DATA_ZONE_START_PSN = 0x030000
SCRAMBLER_SEEDS = np.array([
    0x0001, 0x5500, 0x0002, 0x2A00, 0x0004, 0x5400, 0x0008, 0x2800,
    0x0010, 0x5000, 0x0020, 0x2001, 0x0040, 0x4002, 0x0080, 0x0005,
], dtype=np.uint16)


def _make_field():
    exp = np.zeros(510, np.uint8)
    log = np.zeros(256, np.int16)
    value = 1
    for i in range(255):
        exp[i] = value; log[value] = i
        value <<= 1
        if value & 256:
            value ^= 0x11D
    exp[255:] = exp[:255]
    mul = exp[log[:, None] + log[None, :]]
    mul[0] = 0; mul[:, 0] = 0
    return exp, mul


_EXP, _MUL = _make_field()


@lru_cache(maxsize=3)
def _generator(parity_bytes):
    g = np.array([1], np.uint8)
    for i in range(parity_bytes):
        p = np.zeros(len(g) + 1, np.uint8)
        p[:-1] ^= g
        p[1:] ^= _MUL[g, _EXP[i]]
        g = p
    return g


def rs_parity(data, parity_bytes):
    """Systematic RS parity, GF(256), roots alpha**0 through alpha**(n-1).

    Batch dimensions arbitrary; input words run along the final axis.
    """
    data = np.asarray(data, dtype=np.uint8)
    out = np.zeros(data.shape[:-1] + (parity_bytes,), np.uint8)
    g = _generator(parity_bytes)[1:]
    for j in range(data.shape[-1]):
        feedback = data[..., j] ^ out[..., 0]
        out[..., :-1] = out[..., 1:]
        out[..., -1] = 0
        out ^= _MUL[feedback[..., None], g]
    return out


def rs_syndromes(words, parity_bytes):
    """Evaluate a completed RS word at the generator roots (Horner form)."""
    words = np.asarray(words, dtype=np.uint8)
    result = np.zeros(words.shape[:-1] + (parity_bytes,), np.uint8)
    roots = _EXP[:parity_bytes]
    for j in range(words.shape[-1]):
        result = _MUL[result, roots] ^ words[..., j, None]
    return result


def _edc_table():
    table = np.arange(256, dtype=np.uint32) << 24
    for _ in range(8):
        high = table >> 31
        table = (table << 1) ^ (high * np.uint32(0x80000011))
    return table


_EDC_TABLE = _edc_table()


def edc(data):
    """MSB-first remainder of data*x**32 / (x**32+x**31+x**4+1).

    No reflection, initialization or final XOR. Returns a uint32 per input row.
    The on-disc representation is most significant byte first.
    """
    data = np.asarray(data, dtype=np.uint8)
    crc = np.zeros(data.shape[:-1], np.uint32)
    for j in range(data.shape[-1]):
        crc = (crc << 8) ^ _EDC_TABLE[((crc >> 24) ^ data[..., j]).astype(np.uint8)]
    return crc


@lru_cache(maxsize=1)
def scramble_sequences():
    """Table 4 + Figure 26; S0 is the seed's low byte before any shift."""
    state = SCRAMBLER_SEEDS.copy()
    seq = np.empty((16, PAYLOAD_BYTES), np.uint8)
    for i in range(PAYLOAD_BYTES):
        seq[:, i] = state & 255
        for _ in range(8):
            feedback = ((state >> 14) ^ (state >> 10)) & 1
            state = ((state << 1) | feedback) & 0x7FFF
    seq.flags.writeable = False
    return seq


def scramble_payload(payload, lba):
    """Scramble OR unscramble a (...,2048) payload with broadcastable LBAs.

    Input to this function is logical host data, not the full 2064-byte frame.
    XOR is its own inverse; use it to precompensate a desired scrambled payload.
    """
    data = np.asarray(payload, dtype=np.uint8)
    if data.shape[-1] != PAYLOAD_BYTES:
        raise ValueError('DVD payload must have 2048 bytes on the final axis')
    ids = ((np.asarray(lba, dtype=np.int64) + DATA_ZONE_START_PSN) >> 4) & 15
    return data ^ scramble_sequences()[ids]


def data_frames(payload, lbas, *, scrambled=True):
    """Build (...,2064) DVD-R data-zone frames, before or after scrambling.

    Reserved six bytes are zero for the non-video, non-protected host data here.
    Header and EDC are never scrambled. EDC covers the unscrambled main data.
    """
    data = np.asarray(payload, dtype=np.uint8)
    if data.shape[-1] != PAYLOAD_BYTES:
        raise ValueError('DVD payload must have 2048 bytes on the final axis')
    lbas = np.broadcast_to(np.asarray(lbas, np.int64), data.shape[:-1])
    psn = lbas + DATA_ZONE_START_PSN
    if np.any(lbas < 0) or np.any(psn > 0xFFFFFF):
        raise ValueError('DVD data-zone sector number is outside 24-bit PSN range')
    frame = np.zeros(data.shape[:-1] + (DATA_FRAME_BYTES,), np.uint8)
    frame[..., 1] = (psn >> 16) & 255
    frame[..., 2] = (psn >> 8) & 255
    frame[..., 3] = psn & 255
    frame[..., 4:6] = rs_parity(frame[..., :4], 2)
    frame[..., 12:2060] = data
    crc = edc(frame[..., :2060])
    for j in range(4):
        frame[..., 2060+j] = (crc >> (24-8*j)) & 255
    if scrambled:
        frame[..., 12:2060] = scramble_payload(data, lbas)
    return frame


def encode_ecc_blocks(payload, start_lbas):
    """(B,16,2048) host data -> (B,16,2366) recording-frame bytes.

    Every first LBA must be a multiple of 16. Blocks are independent, so callers
    can use small batches without retaining a state or losing boundary bytes.
    """
    data = np.asarray(payload, dtype=np.uint8)
    if data.ndim != 3 or data.shape[1:] != (16, 2048):
        raise ValueError('payload must have shape (B,16,2048)')
    start_lbas = np.broadcast_to(np.asarray(start_lbas, np.int64), (len(data),))
    if np.any(start_lbas % 16):
        raise ValueError('ECC blocks must start at a multiple of 16 sectors')
    lbas = start_lbas[:, None] + np.arange(16)
    base = data_frames(data, lbas).reshape(-1, 192, 172)
    outer = np.concatenate([base, rs_parity(base.transpose(0,2,1), 16).transpose(0,2,1)],axis=1)
    ecc = np.concatenate([outer, rs_parity(outer, 10)], axis=2)
    recorded = np.empty((len(data),16,13,182), np.uint8)
    recorded[:,:,:12,:] = ecc[:,:192,:].reshape(-1,16,12,182)
    recorded[:,:,12,:] = ecc[:,192:,:]
    return recorded.reshape(-1,16,RECORDING_FRAME_BYTES)


def encode_ecc_block(payload, start_lba):
    """Single-block convenience wrapper: (16,2048) -> (16,2366)."""
    data = np.asarray(payload, dtype=np.uint8)
    if data.shape != (16,2048):
        raise ValueError('payload must have shape (16,2048)')
    return encode_ecc_blocks(data[None], [start_lba])[0]


def deinterleave_recording_frames(recorded):
    """(...,16,2366) -> (...,208,182) ECC matrix, for audits/readback models."""
    recorded = np.asarray(recorded, dtype=np.uint8)
    if recorded.shape[-2:] != (16,2366):
        raise ValueError('recorded shape must end in (16,2366)')
    rows = recorded.reshape(recorded.shape[:-2]+(16,13,182))
    return np.concatenate([rows[...,:,:12,:].reshape(recorded.shape[:-2]+(192,182)),
                           rows[..., :,12,:]], axis=-2)


# Systematic payload byte j is data-frame byte j+12. PI appends 10 bytes per
# row, so it changes byte positions but not row or column of the original data.
_PAYLOAD_ROW, _PAYLOAD_COL = np.divmod(np.arange(2048)+12,172)
PAYLOAD_RECORDING_OFFSETS = _PAYLOAD_ROW*182+_PAYLOAD_COL
# Each 182-byte recording row has two sync frames: 32 sync bits + 91*16 data.
PAYLOAD_CHANNEL_OFFSETS = (_PAYLOAD_ROW*2976 + (_PAYLOAD_COL//91)*1488 + 32 +
                           (_PAYLOAD_COL%91)*16)


def payload_offsets(*, channel=False, centers=False):
    """Per-sector payload offsets, in recording bytes or EFMPlus channel clocks."""
    if centers and not channel:
        raise ValueError('centers applies only to channel-clock positions')
    offsets = PAYLOAD_CHANNEL_OFFSETS if channel else PAYLOAD_RECORDING_OFFSETS
    return offsets.astype(float)+7.5 if centers else offsets.copy()
