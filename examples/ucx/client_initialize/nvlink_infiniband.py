from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize


def main():
    initialize(
        enable_tcp_over_ucx=True,
        enable_nvlink=True,
        enable_infiniband=True,
        enable_rdmacm=True,
        net_devices="mlx5_0:1",
    )
    client = Client("<scheduler_address>:8786") # can also provide cluster

    # your client code here

if __name__ == "__main__":
    main()