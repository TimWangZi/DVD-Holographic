"""Small unbootable ISO9660 volume with a fixed-extent hologram payload file.

The file system lives before sector 32. Burning this ISO preserves each 2048-byte
logical sector; rebuilding an image from its files would destroy the mapping.
"""
from pathlib import Path
import struct

SECTOR=2048
PAYLOAD_START=32


def both16(n): return struct.pack('<H',n)+struct.pack('>H',n)
def both32(n): return struct.pack('<I',n)+struct.pack('>I',n)


def directory_record(extent,size,name,is_dir=False):
    length=33+len(name)+(len(name)%2==0)
    r=bytearray(length); r[0]=length; r[2:10]=both32(extent); r[10:18]=both32(size)
    r[18:25]=bytes([126,11,16,0,0,0,0]); r[25]=2 if is_dir else 0
    r[28:32]=both16(1); r[32]=len(name); r[33:33+len(name)]=name
    return r


def create_image(path,sectors,*,volume_id='DVD_R_TETRAHEDRON',description=None):
    with Path(path).open('wb') as f:
        f.truncate(sectors*SECTOR)
    write_metadata(path,sectors,volume_id=volume_id,description=description)


def write_metadata(path,sectors,*,volume_id='DVD_R_TETRAHEDRON',description=None):
    """Write fixed metadata sectors, leaving the optical file extent unchanged."""
    if sectors<=PAYLOAD_START or (sectors-PAYLOAD_START)*SECTOR>=2**32:
        raise ValueError('ISO9660 hologram extent must fit one 32-bit file length')
    text=(b'DVD-R TETRAHEDRON HOLOGRAM EXPERIMENT\r\n'
          b'Burn the original ISO sector-for-sector to single-layer DVD-R.\r\n'
          b'Do not rebuild an ISO from these files. HOLOGRAM.BIN is optical payload.\r\n'
          b'This is simulated, uncalibrated optical data, not a DVD video.\r\n')
    if description is not None:text=description.encode('ascii')
    if len(text)>SECTOR or len(volume_id.encode('ascii'))>32:
        raise ValueError('ISO label/README exceeds reserved metadata space')
    pvd=bytearray(SECTOR); pvd[:7]=b'\x01CD001\x01'
    pvd[8:40]=b'CDHOLO'.ljust(32); pvd[40:72]=volume_id.encode('ascii').ljust(32)
    pvd[80:88]=both32(sectors); pvd[120:124]=both16(1); pvd[124:128]=both16(1)
    pvd[128:132]=both16(SECTOR); pvd[132:140]=both32(10)
    pvd[140:144]=struct.pack('<I',18); pvd[148:152]=struct.pack('>I',19)
    pvd[156:190]=directory_record(20,SECTOR,b'\x00',True)
    pvd[190:813]=b' '*623
    for offset in (813,830,847,864): pvd[offset:offset+17]=b'0000000000000000\x00'
    pvd[813:830]=b'2026111600000000\x00'; pvd[881]=1
    term=bytearray(SECTOR); term[:7]=b'\xffCD001\x01'
    little=bytes([1,0])+struct.pack('<I',20)+struct.pack('<H',1)+b'\x00\x00'
    big=bytes([1,0])+struct.pack('>I',20)+struct.pack('>H',1)+b'\x00\x00'
    root=b''.join([directory_record(20,SECTOR,b'\x00',True),directory_record(20,SECTOR,b'\x01',True),
                   directory_record(PAYLOAD_START,(sectors-PAYLOAD_START)*SECTOR,b'HOLOGRAM.BIN;1'),
                   directory_record(21,len(text),b'README.TXT;1')])
    with Path(path).open('r+b') as f:
        for lba,data in [(16,pvd),(17,term),(18,little),(19,big),(20,root),(21,text)]:
            f.seek(lba*SECTOR); f.write(data)
