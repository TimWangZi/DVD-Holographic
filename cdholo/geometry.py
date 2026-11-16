"""All lengths in mm. Frame zero is the configurable program-area anchor."""
import numpy as np

CHANNEL_RATE = 4_321_800
FRAME_RATE = 7350


class Spiral:
    def __init__(self, cfg):
        self.r0 = cfg['program_start_radius_mm']
        self.pitch = cfg['track_pitch_um'] * 1e-3
        self.velocity = cfg['linear_velocity_m_s'] * 1000
        self.angle = np.deg2rad(cfg['start_angle_deg'])
        self.offset = cfg['frame_offset']
        self.direction = cfg['spiral_direction']
        self.bit_mm = self.velocity / CHANNEL_RATE

    def coordinates(self, frame, bit=0):
        length = ((np.asarray(frame,dtype=np.float64) + self.offset) * 588 + bit) * self.bit_mm
        r = np.sqrt(self.r0**2 + self.pitch * length / np.pi)
        theta = self.angle + self.direction * 2*np.pi*(r-self.r0)/self.pitch
        return r*np.cos(theta), r*np.sin(theta)

    def frame_at_radius(self, radius):
        return np.pi*(np.asarray(radius)**2-self.r0**2)/self.pitch/self.bit_mm/588-self.offset

    def local_coordinates(self, frame, bit, patch):
        x,y = self.coordinates(frame, bit)
        a = np.deg2rad(patch['angle_deg'])
        return -x*np.sin(a)+y*np.cos(a), x*np.cos(a)+y*np.sin(a)-patch['center_radius_mm']

    def patch_frames(self, patch, margin=2):
        """Find short track intervals crossing the square, without scanning whole disc."""
        r = patch['center_radius_mm']; w,h = patch['size_mm']
        # Allow the corner's extra radius; exact square membership checked downstream.
        rmin = r-h/2
        rmax = np.hypot(r+h/2,w/2)
        a = np.deg2rad(patch['angle_deg'])
        k0 = int(np.floor((rmin-self.r0)/self.pitch))-2
        k1 = int(np.ceil((rmax-self.r0)/self.pitch))+2
        phase = ((a-self.angle)*self.direction/(2*np.pi)) % 1
        radii = self.r0 + self.pitch*(np.arange(k0,k1+1)+phase)
        centers = self.frame_at_radius(radii)
        half = int(np.ceil(w/self.bit_mm/588/2))+margin
        frames = (np.floor(centers).astype(np.int64)[:,None] + np.arange(-half,half+1)).ravel()
        return np.unique(frames[frames>=0])
