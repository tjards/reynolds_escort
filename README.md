# Reynolds Rules of Flocking This project implements Reynolds Rules of Flocking (or *Boids*), which is based on a balance between three steering forces:- **Cohesion**: the tendency to steer towards the center of mass of neighbouring agents- **Alignment**: the tendency to steer in the same direction as neighbouring agents- **Separation**: the tendency to steer away from neighbouring agents Reynolds (1987) did not provide equations for the steering forces above, so we have taken some artistic liberty and had a bit of fun. In the results section, you will see variations on how "neighbours" are defined. We demonstrate free flocking and flocking to escort (i.e. following a reference). A more formal definition and analysis of flocking was provided by Olfati-Saber (2006), which we have implemented [here](https://github.com/tjards/flocking_network).## CitingThe code is opensource but, if you reference this work in your own reserach, please cite me. I have provided an example bibtex citation below:`@techreport{Jardine-2021,  title={Reynolds Rules of Flocking for Tactical Escort'},  author={Jardine, P.T.},  year={2021},  institution={Royal Military College of Canada, Kingston, Ontario},  type={Technical Report},}`Alternatively, you can cite any of my related papers, which are listed in [Google Scholar](https://scholar.google.com/citations?hl=en&user=RGlv4ZUAAAAJ&view_op=list_works&sortby=pubdate).## ReferencesThis work is related to the following research:1. Craig W. Reynolds, "Flocks, herds and schools: A distributed behavioral model", *ACM SIGGRAPH Computer Graphics* vol. 21, pp. 25–34, 1987.2. Reza Olfati-Saber, ["Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory"](https://ieeexplore.ieee.org/document/1605401), *IEEE Transactions on Automatic Control*, Vol. 51 (3), Mar 2006.# ResultsBelow are several animated plots showing the behaviour of the swarm.## Fixed cohesion distance (5m)<p float="center">  <img src="https://github.com/tjards/reynolds_escort/blob/master/Figs/escort_7_agents_min5.gif" width="45%" />  <img src="https://github.com/tjards/reynolds_escort/blob/master/Figs/escort_5m.gif" width="45%"  /></p>## Fixed cohesion distance (10m)<p float="center">        <img src="https://github.com/tjards/reynolds_escort/blob/master/Figs/range_10m.gif" width="70%"  /></p>## Cohesion with nearest agent only<p float="center">        <img src="https://github.com/tjards/reynolds_escort/blob/master/Figs/nearest_1.gif" width="70%"  /></p>## Cohesion with nearest two agents <p float="center">  <img src="https://github.com/tjards/reynolds_escort/blob/master/Figs/escort_nearest_2_noZoom.gif" width="45%" />      <img src="https://github.com/tjards/reynolds_escort/blob/master/Figs/escort_nearest_2.gif" width="45%"  /></p>## Cohesion with nearest five agents <p float="center">        <img src="https://github.com/tjards/reynolds_escort/blob/master/Figs/nearest_5.gif" width="70%" /></p> 