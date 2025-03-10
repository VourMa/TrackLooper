#!/bin/bash

# HiPerGator module setup for cuda
module load cuda/11.4.3 git
# module use ~/module
# module load root/6.22.08

###########################################################################################################
# Setup environments
###########################################################################################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/code/rooutil/thisrooutil.sh

export SCRAM_ARCH=slc7_amd64_gcc10
export CMSSW_VERSION=CMSSW_12_5_0_pre2
export CUDA_HOME=${HPC_CUDA_DIR}

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src
eval `scramv1 runtime -sh`
cd - > /dev/null
echo "Setup following ROOT.  Make sure it's slc7 variant. Otherwise the looper won't compile."
which root

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$DIR:$LD_LIBRARY_PATH
export PATH=$DIR/bin:$PATH
export PATH=$DIR/efficiency/bin:$PATH
export PATH=$DIR/efficiency/python:$PATH
export TRACKLOOPERDIR=$DIR
export TRACKINGNTUPLEDIR=/blue/p.chang/p.chang/data/lst/CMSSW_12_2_0_pre2
export PIXELMAPDIR=/blue/p.chang/p.chang/data/lst/pixelmap_neta20_nphi72_nz24_ipt2
export LSTOUTPUTDIR=.
export LSTPERFORMANCEWEBDIR=/blue/p.chang/users/phchang/public_html/LSTPerformanceWeb

###########################################################################################################
# Validation scripts
###########################################################################################################

# List of benchmark efficiencies are set as an environment variable
export LATEST_CPU_BENCHMARK_EFF_MUONGUN=
export LATEST_CPU_BENCHMARK_EFF_PU200=
#eof
