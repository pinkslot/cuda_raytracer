Fork from cuda raytracing tutorial project https://github.com/straaljager/GPU-path-tracing-tutorial-3.

* Volume scettering was added.
* Phase point was extended with time coordinate to maintain nonstationry scene.
* Different branching methods was implemented. Main path tracing loop was replaced to stack emulation for this purpose.
* Sphere object definition was reworked. Now they have separate structure to describe its inner media and boundary light source.
