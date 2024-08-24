# Visual Tracking
This repository records the proposed visual tracking methods based on [PyTracking](https://github.com/visionml/pytracking) Library

## Trackers
Here is a list of the improved trackers.  

[Raw tracking results and trained models](https://drive.google.com/drive/folders/182NbsBrVR9PICR9aSkb2IhUDvrlSsTDT?usp=sharing)
* CAT: Color Attention Tracking with Score Matching, International Journal of Machine Learning and Cybernetics, 2024. [[Paper]](https://www.doi.org/10.1007/s13042-024-02316-y)
* FuDiMP: Attention Fusion and Target-Uncertain Detection for Discriminative Tracking, Knowledge-Based Systems, 2023. [[Paper]](https://doi.org/10.1016/j.knosys.2023.110860)
* EnDiMP: Enhancing Discriminative Appearance Model for Visual Tracking, Expert Systems With Applications, 2023. [[Paper]](https://doi.org/10.1016/j.eswa.2023.119670)
* RVT: Exploring Reliable Visual Tracking via Target Embedding Network, Knowledge-Based Systems, 2022. [[Paper]](https://doi.org/10.1016/j.knosys.2022.108584)  
* OUPT: Learning Object-Uncertainty Policy for Visual Tracking, Information Sciences, 2022. [[Paper]](https://doi.org/10.1016/j.ins.2021.09.002)
* VSLT: Variable Scale Learning for Visual Object Tracking, Journal of Ambient Intelligence and Humanized Computing, 2021. [[Paper]](https://doi.org/10.1007/s12652-021-03469-2)  

###  Highlights
Moreover, we also integrate [got10k](pytracking/external/got10k) and [pysot_toolkit](pytracking/external/pysot_toolkit) for evaluation.  

The [refine_modules](pytracking/external/refine_modules) of AlphaRefine tracker also is attached to adapt for bounding-box to mask output. 

ECO, DiMP, TOMP, D3S, SiamFC, TranT, and TrDiMP trackers are seamlessly integrated into VisTrack project. 

## Findings
* The UAV123 results of pysot and pytracking are consistent, which are lower than the results of got10k toolkit. 
  From the results reported in the original conference papers, the authors probably use the evaluation results given by got10k.

## Contact
* Xuedong He (email:hexuedong@zjnu.edu.cn)   
Feel free to contact me if you have additional questions. 

