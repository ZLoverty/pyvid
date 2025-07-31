# pyvid: 3D animation data visualization tool

`pyvid` is a 3D animation cdata visualization tool based on `pyvista`. This tool works best for scenes that have relatively few actors. The idea is to input time series data, especially trajectories, to the scene along with the actor. The tool features:

- custom camera motions defined by keyframes
- custom playback speed (can vary over time) useful for creating slow close-up
- real time tuning of the script in Jupyter notebook

## Installation

```
git clone https://github.com/ZLoverty/pyvid.git
```

## Basic guide

A typical workflow is to create a Jupyter notebook and create a scene using

```
scene = Scene()
```

Then, add actors and data to the scene. 

Then, define camera motion and time mesh. 

Lastly, use 

```
scene.play()
```

to preview the animation.

Lastly, use 

```
scene.play(record=True, save_folder=".") 
```

to save the animation in a video file. 

## Versions

- 0.1.0: First version.

## Enhancedment proposals (PVEP)