#!/bin/bash

set -xo pipefail

# load aliases like module
source /etc/profile

module load cuda/10.2.89.0
#export PATH=/gpfs/fs1/bzaitlen/miniconda3/bin:$PATH
CONDA_ROOT=/gpfs/fs1/bzaitlen/miniconda3
if test -f ~/.profile; then
    source ~/.profile
fi
source $CONDA_ROOT/etc/profile.d/conda.sh
ENV=`date +"%Y%m%d-nightly-0.17"`

srun -N1 create-env.sh
# Environment variables to enable GPUs, InfiniBand, NVLink
# These are read by the scheduler and client script
conda activate $ENV
which python

# Each worker uses all GPUs on its node
# Make all NICs available to the scheduler. "--net-devices auto" overrides this
# for workers: each subprocess is assigned the best NIC for its GPU.

# Prepare output directory
JOB_OUTPUT_DIR=slurm-dask-`date +"%Y%m%d"`
mkdir $JOB_OUTPUT_DIR

# Start a single scheduler on node 0 of the allocation
   DASK_UCX__CUDA_COPY=True \
   DASK_UCX__TCP=True \
   DASK_UCX__NVLINK=True \
   DASK_UCX__INFINIBAND=True \
   DASK_RMM__POOL_SIZE=0.2GB \
   DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s" \
   DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s" \
   DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s" \
   DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s" \
   DASK_DISTRIBUTED__WORKER__MEMORY__Terminate="False" \
   UCX_NET_DEVICES=mlx5_0:1 \
   UCX_SOCKADDR_TLS_PRIORITY=sockcm \
   srun -N 1 -n 1 \
   python -m distributed.cli.dask_scheduler \
  --protocol ucx \
  --interface ib0 \
  --scheduler-file "$JOB_OUTPUT_DIR/cluster.json" &

# Wait for the scheduler to start
sleep 10
SCHED_ADDR="$(python -c "
import json
with open('$JOB_OUTPUT_DIR/cluster.json') as f:
  print(json.load(f)['address'])
")"
# Start one worker per node in the allocation (one process started per GPU)
for HOST in `scontrol show hostnames "$SLURM_JOB_NODELIST"`; do
  DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s" \
  DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s" \
  DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s" \
  DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s" \
  DASK_DISTRIBUTED__WORKER__MEMORY__Terminate="False" \
  srun -N 1 -w "$HOST" python -m dask_cuda.cli.dask_cuda_worker \
    --enable-tcp-over-ucx \
    --enable-nvlink \
    --enable-infiniband \
    --net-devices="auto" \
    --rmm-pool-size 13G \
    --local-directory "$JOB_OUTPUT_DIR/$HOST" \
    --scheduler-file "$JOB_OUTPUT_DIR/cluster.json" &
done
# Wait for the workers to start
sleep 5

# Execute the client script on node 0 of the allocation
# The client script should shut down the scheduler before exiting
#"$CONDA_PREFIX/lib/python3.8/site-packages/dask_cuda/benchmarks/local_cudf_merge.py" \
echo "Client start: $(date +%s)"
   DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s" \
   DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="100s" \
   DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s" \
   DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s" \
   DASK_DISTRIBUTED__WORKER__MEMORY__Terminate="False" \
   UCX_SOCKADDR_TLS_PRIORITY=sockcm \
   UCX_NET_DEVICES=mlx5_0:1 \
   UCX_TLS=tcp,cuda_copy,cuda_ipc,rc,sockcm \
   DASK_RMM__POOL_SIZE=0.5GB \
   UCX_LOG_LEVEL=ERROR \
   srun -N 1 -n 1 \
   python \
   "/home/bzaitlen/GitRepos/dask-cuda/dask_cuda/benchmarks/local_cudf_merge.py" \
  --scheduler-address "$SCHED_ADDR" \
  -c 50_000_000 \
  --frac-match 1.0 \
  --protocol ucx \
  --disable-rmm-pool \
  --markdown \
  --all-to-all \
  --runs 10 \
  --profile "$JOB_OUTPUT_DIR/$ENV-dask-cudf-merge-profile.html" \
  --plot "$JOB_OUTPUT_DIR"  > $JOB_OUTPUT_DIR/raw_data.txt

echo "Client done: $(date +%s)"
# Wait for the cluster to shut down gracefully
sleep 2

# Upload results
srun -N 1 -n 1 \
   python \
   "/home/bzaitlen/GitRepos/dask-cuda/dask_cuda/benchmarks/publish_benchmark.py"
