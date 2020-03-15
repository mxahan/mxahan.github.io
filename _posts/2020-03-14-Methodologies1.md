# Optical Flow
For frames in video, we assume object moves but the intensity of pixel remains same. 
$$ I(x, y, t) = I(x +dx, y+dy, t+dy) $$ 
Now using taylors formula 
$$I(x +dx, y+dy, t+dy) = I(x, y, t)+ \frac{\delta I}{\delta t} \delta x+ \frac{\delta I}{\delta t}\delta y +\frac{\delta I}{\delta t}\delta t + ...$$ 
Combining the earlier two gives,
$$\frac{\delta I}{\delta t} \delta x+ \frac{\delta I}{\delta t}\delta y +\frac{\delta I}{\delta t}\delta t = 0 $$
$$\frac{\delta I}{\delta t} u+ \frac{\delta I}{\delta t}v +\frac{\delta I}{\delta t} = 0 $$
This show relation between image gradients alone x, y and time axis. The unknowns are *u* and *v*. This requires methods like Mean shift color histogram tracking, Lucas-Kanade methods. It's an optimization problem. 

A distinction to keep in mind for recovering motion. 
1. *Feature-tracking;* Extract visual features and track them 
2. *Optical flow;*  Recover image motion at pixel from spatio-temporal image brightness variations (the brightness assumption, small motion and spatial coherence should maintain). 

Solving equation, modified and matrix form from the earlier equation. 

$$Au = b$$
$$A^TAu = A^Tb$$
