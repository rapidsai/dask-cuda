from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize


def main():
    initialize(
        enable_tcp_over_ucx=True,
        net_devices="enp1s0f0",
    )
    client = Client("<scheduler_address>:8786") # can also provide cluster

    # your client code here

if __name__ == "__main__":
    main()