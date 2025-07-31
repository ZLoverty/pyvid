from pyvistaqt import BackgroundPlotter
import pyvista as pv
import numpy as np
from scipy.interpolate import interp1d
from PyQt5.QtCore import QTimer
from pathlib import Path

class Scene(BackgroundPlotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clear_all()

    def Sphere(self, data, **kwargs):
        """Add mesh as actor to scene. 

        Parameters
        ----------
        mesh : pyvista.DataSet
            The mesh to be added.
        data : dict
            Trajectory data, must contain 't' and 'x'.
        
        Returns
        -------
        actor : pv.Actor
            The actor object.
        """
        t, x = self._read_data(data)

        # update time range
        self.T = max(self.T, t.max())
        self.t0 = min(self.t0, t.min())

        # add actor
        actor = self.add_mesh(pv.Sphere(**kwargs))
        actor_data = (actor, {"t": t, "x": x})
        self.actor_list.append(actor_data)

        return actor

    def Box(self, data, **kwargs):
        """Add mesh as actor to scene. 

        Parameters
        ----------
        mesh : pyvista.DataSet
            The mesh to be added.
        data : dict
            Trajectory data, must contain 't' and 'x'.
        
        Returns
        -------
        actor : pv.Actor
            The actor object.
        """
        t, x = self._read_data(data)

        # update time range
        self.T = max(self.T, t.max())
        self.t0 = min(self.t0, t.min())

        # add actor
        actor = self.add_mesh(pv.Box(**kwargs))
        actor_data = (actor, {"t": t, "x": x})
        self.actor_list.append(actor_data)

        return actor
    
    def _read_data(self, data):
        """Validate the data: (i) contains 'x' and 't', (ii) 'x' has shape (N, 3), (iii) t.shape[0] = x.shape[0].
        
        Parameters
        ----------
        data : dict
            Trajectory data containing "t" and "x".
        """
        if not all(k in data for k in ["t", "x"]):
            raise ValueError("Input data dictionary must contain 't' and 'x'.")
        
        t = np.asarray(data["t"])
        x = np.asarray(data["x"])

        # ensure position data shape
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("The position data shape must be (N, 3).")
        
        # ensure t and x are the same length
        assert x.shape[0] == t.shape[0]

        return t, x

    def Glyph(self, data, name, factor=1.0, **kwargs):
        """Add glyph actor to the scene.
        
        Parameters
        ----------
        data : dict
            Glyph data, containing "t", "x" and "v". "x" and "v" are both (M, N, 3), where M is the number of time steps and N is the number of points.
        name : str
            Name used to set the actor name, in order to update the scene.
        factor : float
            Scale factor of glyph length.
        
        Returns
        -------
        actor : pv.Actor
            Glyph actor object. Can be used to set actor properties.
        """
        t = np.asarray(data["t"])
        x = np.asarray(data["x"])
        v = np.asarray(data["v"])
        
        # update time range
        self.T = max(self.T, t.max())
        self.t0 = min(self.t0, t.min())

        # create glyph PolyData
        glyph = self._gen_glyph(x[0], v[0], factor=factor)

        # add actor
        actor = self.add_mesh(glyph, name=name, **kwargs)
        actor_data = (actor, {"t": t, "x": x, "v": v}, name, factor, kwargs)
        self.glyph_list.append(actor_data)
        return actor

    def _gen_glyph(self, x, v, factor=1.0):
        """Generate glyph from points and velocity.
        
        Parameters
        ----------
        x : ndarray
            Position data of (N, 3).
        v : ndarray
            Velocity data of (N, 3).

        Returns
        -------
        glyph : pyvista.PolyData
            Glyph object that can be added to `pyvista.Plotter` by `add_mesh`.
        """
        if np.all(np.isnan(v)):
            v = np.zeros_like(v)
        grid = pv.PolyData(x)
        grid["v"] = v
        glyph = grid.glyph(orient="v", scale="v", factor=factor)
        return glyph
    
    def set_time(self, t):
        """Set the time points to prepare the video. Update the 'data_interp' field with interpolated data.
        
        Parameters
        ----------
        t : array_like[float]
            Time sequence to evaluate the object positions.
        """
        self.t = t

    def set_camera_keyframes(self, keyframes):
        """Set camera keyframes to achieve effects like slow close-up.
        
        Parameters
        ----------
        keyframes : list
            Keyframe list, containing tuples of (time, camera_pos). camera_pos is defined as an ndarray of (3, 3).
        """
        self.camera_keyframes = keyframes

    def play(self, fps=30, playback=1.0, t_range=None, record=False, save_folder=None):
        """Play animation.
        
        Parameters
        ----------
        fps : float
            Frame rate of the animation.
        playback : float
            Playback speed of the animation. The total number of frames is calculated as 
            
                nFrame = t_total * fps / playback

            This is needed only if the time sequence is not set. Otherwise, the input time sequence will be used and playback will be ignored. 
        t_range : Sequence
            [tmin, tmax], the animation is limited by this range.  
        record : bool
            Whether to generate a video. Default to False.
        
        """
        
        if self.t is not None: # if self.t is already set by self.set_time()
            t = self.t
        else: # if not set
            print(f"WARNING: Time sequence is not set, infer from actor data: {self.t0:.4f} - {self.T:.4f}. Use `set_time()` to set custom time sequence.")
            t_total = self.T - self.t0
            nFrame = int(t_total * fps / playback)
            if nFrame <= 1:
                raise ValueError("Too few frames! Check total time, fps and playback speed!")
            t = np.linspace(self.t0, self.T, nFrame)

        # apply time range to play a portion of the animation
        if t_range: # if t_range is set, determine the indices 
            if len(t_range) == 2:
                ind = np.logical_and(t>t_range[0], t<=t_range[1])
            else:
                raise ValueError("t_range must be a tuple or list of two floats.")
        else: # if t_range is not set, use all True indices
            ind = np.ones_like(t).astype(bool)
        
        t = t[ind]
    
        # interpolate data to this time range
        actor_list_interp = []
        for actor, data in self.actor_list:
            interp_data = {}
            for kw in data:
                if kw == "t":
                    interp_data[kw] = t
                else:
                    f = interp1d(data["t"], data[kw], axis=0, bounds_error=False)
                    interp_data[kw] = f(t)
            actor_list_interp.append((actor, interp_data)) 
        
        # interpolate glyph data
        glyph_list_interp = []
        for actor, data, name, factor, kwargs in self.glyph_list:
            interp_data = {}
            for kw in data:
                if kw == "t":
                    interp_data[kw] = t
                else:
                    f = interp1d(data["t"], data[kw], axis=0, bounds_error=False)
                    interp_data[kw] = f(t)
            glyph_list_interp.append((actor, interp_data, name, factor, kwargs))

        # interpolate camera motion    
        t_cp = np.asarray([float(t_) for t_, _ in self.camera_keyframes])
        cp = np.stack([cp_ for _, cp_ in self.camera_keyframes])
        f = interp1d(t_cp, cp, axis=0, bounds_error=False)
        camera_positions = f(t)

        # if record to file, create folders if necessary
        if record:
            try:
                save_folder = Path(save_folder).expanduser().resolve()
            except TypeError:
                raise TypeError("Set save_folder to a dir string.")
            tmp_folder = save_folder / "_tmp"
            if tmp_folder.exists() == False:
                tmp_folder.mkdir(parents=True, exist_ok=True)

        # animate the scene using QTimer
        delay = int(1 / fps * 1000)
        step = [0]
        timer = QTimer()
        
        def update():
            
            if step[0] >= len(t)-1:
                timer.stop()
                print(f"Animation complete ({step[0]:d}/{len(t)-1:d}).")
                return
            # update actors
            for actor, data in actor_list_interp:
                new_position = data["x"][step[0]]
                actor.SetPosition(new_position)
            # update glyphs
            for glyph, data, name, factor, kwargs in glyph_list_interp:
                new_x = data["x"][step[0]]
                new_v = data["v"][step[0]]
                glyph = self._gen_glyph(new_x, new_v, factor=factor)
                self.add_mesh(glyph, name=name, **kwargs)
            # update camera positions
            self.camera_position = camera_positions[step[0]]
            self.render() 

            if record:
                self.screenshot(
                    filename = tmp_folder / f"{step[0]:04d}.jpg",
                    return_img = False
                )

            timer.start(delay)
            step[0] += 1
            print(f"t: {step[0]/fps:3.1f} s, step: {step[0]:d}/{len(t)-1:d}", end="\r")
            
        timer.timeout.connect(update)
        timer.start(delay)

    def clear_all(self):
        """Clear actor list and time."""
        try:
            l = getattr(self, "actor_list")
            for actor, _ in l:
                self.remove_actor(actor)
            l = getattr(self, "glyph_list")
            for actor, _, _, _, _ in l:
                self.remove_actor(actor)
        except AttributeError:
            pass
        self.actor_list = []
        self.glyph_list = []
        self.T = 0
        self.t0 = 0
        self.t = None # The time sequence
        self.camera_keyframes = None 