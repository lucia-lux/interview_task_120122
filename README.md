# interview_task_120122
Data analysis for interview on 12/01/2022.

Interview on 12/01/2022.

Task is as follows:

(1) Describe the sample at baseline.
(2) Perform at least one statistical test that shows the effectiveness of the intervention (or lack thereof).
(3) The analysis must contain at least one plot/visualization.

Time:
2-3 hours max

Purpose:
Approach to coding/analysis more important than extent of analysis performed.

Solution:
Use OOP approach. Individual classes for Data preparation/Baseline analysis/Data analysis.
Preparation:
Rename some cols, convert wide to long format (longtdl. data)

Baseline:
Plot univariate distributions
Check whether individual features normally distributed
check whether groups differ on baseline features

Analysis
Use GEEs or LMEMs (as appropriate, depends on distribution and whether we are interested in modelling random effects explicitly (ie population averaged response vs individual differences).
Pick one outcome and build appropriate model. (check bivariate plots).
Model diagnostics - assumption checks
