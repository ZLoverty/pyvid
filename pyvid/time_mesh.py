import numpy as np
from scipy.interpolate import interp1d, CubicSpline

class TimeMesh:
    """Generate time mesh. By default, generate a uniform mesh according to start, end and fps. Optionally, insert key frames for fps (e.g. for slow motion) and interpolate time mesh."""
    def __init__(self, start, end, fps):
        self.start = start
        self.end = end
        self.fps = fps
        self.keyframes = None
    
    def __repr__(self):
        return f"Time mesh in ({self.start:.1f}, {self.end:.1f}), with base frame rate {self.fps:.1f}"
    
    def uniform(self):
        """Uniform time mesh."""
        nFrame = int((self.end - self.start) * self.fps)
        return np.linspace(self.start, self.end, nFrame)
    
    def insert_keyframes(self, keyframes: list):
        """Set keyframes
        
        Parameters
        ----------
        keyframes : list(tuple)
            list of (time, fps)
        """
        for t, _ in keyframes:
            assert(t >= self.start)
            assert(t <= self.end)
        self.keyframes = keyframes
        self.keyframes.append((self.start, self.fps))
        self.keyframes.append((self.end, self.fps))
        self.keyframes = set(self.keyframes)
        self.keyframes = np.array(list(self.keyframes))
        ind = np.argsort(self.keyframes[:, 0])
        self.keyframes = self.keyframes[ind]

    def generate(self, interpolate="cubic"):
        """Generate the time mesh according to the keyframes data: if no data is provided, generate uniform mesh; if keyframe is provided, interpolate the fps function using "cubic" or "linear" method.
        
        Parameters
        ----------
        interploate : str
            "cubic" or "linear"
        """
        if self.keyframes is None:
            return self.uniform()

        match interpolate: 
            case "cubic":
                fps_func = CubicSpline(self.keyframes[:, 0], self.keyframes[:, 1], bc_type="natural")
                # !
            case "linear":
                fps_func = interp1d(self.keyframes[:, 0], self.keyframes[:, 1])
        
        mesh = [self.start]
        t = self.start
        while t < self.end:
            interval = 1 / fps_func(t)
            t += interval
            mesh.append(t)
        return np.array(mesh)