"""The radar: the objects around the car, kept over time (owner's design, 2026-11-22).

A component of its own, wired into the stack like the map box: in go the car's own motion (`ego_state`: speed,
yaw rate, acceleration) and what any object sensor reports (today the camera's class detector, `road_users`;
by the same interface further cameras, the road model's lead and microwave radars); out comes one stream,
`radar`: every object with its position, velocity and acceleration over the ground, its direction of travel,
what it is, whether it stands or moves, how sure that is, and where it will be.

How it is built (each part is a file):

* `ego.py`      **the car's own movement.** The car's pose on the ground (a local frame that starts where the
                radar started) integrated from speed, yaw rate and acceleration, with its history: a
                measurement is placed where the car was when it was taken, not where it is when it arrives.
* `models.py`   **kinematics and dynamics.** The objects live in the ground frame, where "it stands" is simply
                zero velocity whatever the car does. Each object runs several motion hypotheses side by side
                (standing, steady velocity, manoeuvring) and what kind of road user it is bounds what each may
                do: a pedestrian's speed and acceleration are not a car's, a lorry's are not a motorbike's.
* `sensors.py`  **sensors are models.** What a sensor measures as a function of an object's state and the car's
                pose, how good it is, where it can see, how often it misses and how often it invents: the
                camera's boxes (a good bearing, a poor ground range with a bias of its own, or a bearing only),
                a microwave radar (range, azimuth, range rate), a plain position sensor.
* `tracker.py`  **model-based filtering, checked against later frames.** Every object is predicted forward by
                its hypotheses; each later detection is compared with that prediction before it is used. The
                comparison decides which detection belongs to which object, how much each hypothesis is
                believed (an interacting-multiple-model filter), whether the object exists at all, and it is
                kept as the object's own record of how well it has been predicted. Objects have a memory: a
                trajectory, a class vote, an age, and a life that does not end with one missed frame.
* `box.py`      the component's face: messages in, the `radar` message out (car frame of the output time).

The arithmetic of all of it (the motion model, the measurement functions, a scan's association and update, the
objects' upkeep and the message's objects) runs in the native core (chris_integrator/native, the `radar` part),
its only implementation: this package does not import without it (docs/NATIVE.md says how to build it). The
objects' numbers live in one array of the tracker (`tracker.State`), which the native calls read and write; the
Python here is the objects' life, the message's assembly and the glue.
* `replay.py`   a recorded session through the component alone.

The plug-in that wires it into the mixer is `chris_integrator/plugins/radar.py`.
"""
from .box import Radar  # noqa: F401
