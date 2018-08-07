# Pyod_CASCO
This is my intern project in CASCO Shanghai, China.


# Extensible OpenFOAM framework for cluster(Beocat)

This is a handy framework for user to run OpenFOAM application on cluster like Beocat.

![](stlpic.png)

## Prerequisites

* [OpenFOAM]  
* [ParaView]

## Installation 
```sh
git clone https://github.com/paragon520/Analysis-of-Water-Control-Structures-on-Cluster
cd Analysis-of-Water-Control-Structures-on-Cluster
``` 

## Test and run on local machine 

Simply run: 
```sh
./clean
./test 
```
 
## Run on cluster(Beocat)
Before you run on cluster:
1. Put this project on your cluster. 
2. Make sure your cluster server has OpenFOAM module installed and check version with your server administrator.
3. Modify the header code in HPCjob.sh, HPCclean.sh and HPCrecon.sh to make it compatible with your cluster.

Clean your previous work if needed.(Optional)
```sh
bash HPCclean.sh
```

Run your job. (Since Beocat using CentOS Linux servers coordinated by the Slurm job submission and scheduling system, we use sbatch to submit our job.) 
```sh
sbatch HPCjob.sh
```
Reconstruct your result if using mpirun to do parallel computing. 
```sh
bash HPCrecon.sh
```

## View your result 
Open dummy file foam.foam in ParaView to view your simulation result.
```sh
paraview & 
```

## Release History

* 0.0.1
    * Add HPC job submission code.

## Lincese

See the  [LICENSE] file.
 

 [LICENSE]: https://github.com/paragon520/Analysis-of-Water-Control-Structures-on-Cluster/blob/master/LICENSE
 [OpenFOAM]:  https://openfoam.org/download/
[ParaView]: https://www.paraview.org/download/
