
This is an implementation of diffusion model for probabilistic human motion prediction.

We referred to the implementation of ddpm from [this repository](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-) and wanted to say thanks to the author here.

### datasets
[Human3.6M from siMLPe](https://github.com/dulucas/siMLPe)   
```
human3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```
[Human3.6M from gsps](https://github.com/wei-mao-2019/gsps)

[AMASS dataset from official website](https://amass.is.tue.mpg.de/download.php) (note: you should download the SMPL+H G)
```
amass
|-- ACCAD
|-- BioMotionLab_NTroje (aka BMLrub)
|-- CMU
|-- ...
`-- TotalCapture
```

[3DPW dataset from official website](https://virtualhumans.mpi-inf.mpg.de/3DPW/)
```
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```
