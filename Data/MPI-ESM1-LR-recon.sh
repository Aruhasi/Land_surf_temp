#!/bin/bash

    cd /work/mh0033/m301036/josie/LSAT/Data/
    
    pwd
    dir=/work/mh0033/m301036/josie/LSAT/Data/
    
    # dir_o2="/work/mh0033/m301036/LSAT/CMIP6-MPI-M-LR/regrid"
    cdo -r copy ${dir}MPI-ESM1-LR-30run-tas.nc MPI-ESM1-LR-30run-tas.recon.nc
    # cdo remapbil,r180x90 ${dir}GR15_lsm.nc GR15_lsm_regrid.nc