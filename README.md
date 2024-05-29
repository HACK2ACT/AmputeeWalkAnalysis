# AmputeeWalkAnalysis
The goal of this software is to help analysing the walk of amputee toi detect defaults, and help to correct them.

# Licence 
This software is under licence GNU GPL 3. 
As defined in the licence there is no warranty of any kind with usage of this code.

# State and contribution
The source is a preliminary version to be considered as a Proof Of Concept.
If you want to contribute to this proeject of send video in mp4 to test the software, please contact us. 

# Principle 
The principle is to draw a simple skeleton using mediapipe and openCV on the person in order to detect the position of different key part such as ankles, shoulders, etc ... This difference of position for a given point of interest is reported in a time graphs in order to highlight rythm, difference between right and left. There is still some work to do in order to ease analysis and avoid wrong analysis. The list of work is : to normalize positions, to detect local extremum to detect impact on the floor, ... Then in a second time some use of ArUco, ChArUco or Diamond markers or boards from openCV. 
In a nutshell, this is the first step tp developp a quality walking open source software.
If you don't want to wait for the developpement of such a tool, you can find some very powerfull commercial software performing this kind of analysis and much more.

# Why ?
I have initiated this project because I am a fresh amputee. I have a discovered a total new word with a lot of great professionals, and motivated patients. I would like to offer a modest contribution to this community with a simple software just to explore the wlaking quality which is very difficult to hanlde by some patients. 
