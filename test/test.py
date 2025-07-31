from pyvid import Scene, TimeMesh
import numpy as np
import pyvista as pv
from pathlib import Path


self = Scene(toolbar=False, menu_bar=False)

class Helloworld(Scene):
    def construct(self):

        # add bubble
        N = 50
        t = np.linspace(0, 1, N)
        x = np.random.rand(N, 3)
        sphere = self.Sphere({"t": t, "x": x})

        # set time sequence
        tm = TimeMesh(0, 1, 30)
        keyframes = [
            (0.5, 50),
        ]
        tm.insert_keyframes(keyframes)
        t_ = tm.generate("linear")
        self.set_time(t_)

        ## set camera keyframes
        cp_far = np.array([
            [-5.41173497e-06,  2.89114193e-04, -2.87610813e-02],
            [-5.41173497e-06,  2.89114193e-04,  0.00000000e+00],
            [ 1.74524064e-02, -9.99847695e-01,  0.00000000e+00]
        ])
        cp_close = np.array([
            [-5.41173497e-06,  2.89114193e-04, -1.10886419e-02],
            [-5.41173497e-06,  2.89114193e-04,  0.00000000e+00],
            [ 1.74524064e-02, -9.99847695e-01,  0.00000000e+00]
        ])

        keyframes = [
            (0, cp_far),
            (1, cp_close)
        ]
        self.set_camera_keyframes(keyframes)
        self.play()

if __name__=="__main__":
    
    h = Helloworld()
    h.construct()
