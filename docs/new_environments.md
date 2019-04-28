This is a brief guide of the new environments present in this branch

SawyerLego

Thid environemnt consists of tetris like shapes made up from cubic block primitives with the objective being to fit in the tetris peice
into the appropriately shaped hole. This task has three versions with varying levels of difficulty ranging from 2D push and fit to pick and place
to full 3D reorient and place. For these sets of tasks a task completion reward function is already implemented.

SawyerFit

This task involves picking up household objects (loaded in using meshes) reasoning about their shape and then reorienting and inserting them into the holes.
The task completion reward function has been implemented for this and there is a script which processes any mesh into a .xml file which can be loaded in. We still
need to narrow down to a set of meshes that work well for this task.

SawyerClutter

This ivolves a set of procedurally generated toys like objects with randomized sizes. These are initialized clutterred in a bin. and the task involves
sorting the objects. A reward function has not been implemented for this

