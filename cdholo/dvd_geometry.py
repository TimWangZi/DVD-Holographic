"""DVD-R program-area spiral; sector zero is PSN 0x030000 by default."""
import numpy as np

BITS_PER_SECTOR=38688


class DvdSpiral:
    def __init__(self,cfg):
        self.r0=cfg['program_start_radius_mm']
        self.pitch=cfg['track_pitch_um']*1e-3
        self.bit_mm=cfg['channel_bit_length_nm']*1e-6
        self.angle=np.deg2rad(cfg['start_angle_deg'])
        self.offset=cfg['channel_offset_bits']
        self.direction=cfg['spiral_direction']

    def coordinates(self,lba,bit=0):
        length=(np.asarray(lba,dtype=np.float64)*BITS_PER_SECTOR+bit+self.offset)*self.bit_mm
        r=np.sqrt(self.r0**2+self.pitch*length/np.pi)
        theta=self.angle+self.direction*2*np.pi*(r-self.r0)/self.pitch
        return r*np.cos(theta),r*np.sin(theta)

    def sector_at_radius(self,r):
        return (np.pi*(np.asarray(r)**2-self.r0**2)/self.pitch/self.bit_mm-self.offset)/BITS_PER_SECTOR

    def local_coordinates(self,lba,bit,patch):
        x,y=self.coordinates(lba,bit); a=np.deg2rad(patch['angle_deg'])
        return -x*np.sin(a)+y*np.cos(a), x*np.cos(a)+y*np.sin(a)-patch['center_radius_mm']

    def patch_sectors(self,patch):
        r=patch['center_radius_mm']; w,h=patch['size_mm']
        rmin=r-h/2; rmax=np.hypot(r+h/2,w/2)
        phase=((np.deg2rad(patch['angle_deg'])-self.angle)*self.direction/(2*np.pi))%1
        turns=np.arange(int(np.floor((rmin-self.r0)/self.pitch))-2,
                        int(np.ceil((rmax-self.r0)/self.pitch))+3)
        centers=self.sector_at_radius(self.r0+self.pitch*(turns+phase))
        half=int(np.ceil(w/(BITS_PER_SECTOR*self.bit_mm)/2))+1
        sectors=(np.floor(centers).astype(np.int64)[:,None]+np.arange(-half,half+1)).ravel()
        return np.unique(sectors[sectors>=0])


def payload_channel_centers():
    row,col=np.divmod(np.arange(2048)+12,172)
    return row*2976+(col//91)*1488+32+(col%91)*16+7.5


def recording_channel_centers():
    row,col=np.divmod(np.arange(2366),182)
    return row*2976+(col//91)*1488+32+(col%91)*16+7.5
