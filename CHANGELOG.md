# dask-cuda 22.10.00 (Date TBD)

Please see https://github.com/rapidsai/dask-cuda/releases/tag/v22.10.00a for the latest changes to this development branch.

# dask-cuda 22.08.00 (17 Aug 2022)

## 🚨 Breaking Changes

- Fix useless property ([#944](https://github.com/rapidsai/dask-cuda/pull/944)) [@wence-](https://github.com/wence-)

## 🐛 Bug Fixes

- Fix `distributed` error related to `loop_in_thread` ([#963](https://github.com/rapidsai/dask-cuda/pull/963)) [@galipremsagar](https://github.com/galipremsagar)
- Add `__rmatmul__` to `ProxyObject` ([#960](https://github.com/rapidsai/dask-cuda/pull/960)) [@jakirkham](https://github.com/jakirkham)
- Always use versioneer command classes in setup.py ([#948](https://github.com/rapidsai/dask-cuda/pull/948)) [@wence-](https://github.com/wence-)
- Do not dispatch removed `cudf.Frame._index` object ([#947](https://github.com/rapidsai/dask-cuda/pull/947)) [@pentschev](https://github.com/pentschev)
- Fix useless property ([#944](https://github.com/rapidsai/dask-cuda/pull/944)) [@wence-](https://github.com/wence-)
- LocalCUDACluster&#39;s memory limit: `None` means no limit ([#943](https://github.com/rapidsai/dask-cuda/pull/943)) [@madsbk](https://github.com/madsbk)
- ProxyManager: support `memory_limit=None` ([#941](https://github.com/rapidsai/dask-cuda/pull/941)) [@madsbk](https://github.com/madsbk)
- Remove deprecated `loop` kwarg to `Nanny` in `CUDAWorker` ([#934](https://github.com/rapidsai/dask-cuda/pull/934)) [@pentschev](https://github.com/pentschev)
- Import `cleanup` fixture in `test_dask_cuda_worker.py` ([#924](https://github.com/rapidsai/dask-cuda/pull/924)) [@pentschev](https://github.com/pentschev)

## 📖 Documentation

- Switch docs to use common `js` &amp; `css` code ([#967](https://github.com/rapidsai/dask-cuda/pull/967)) [@galipremsagar](https://github.com/galipremsagar)
- Switch `language` from `None` to `&quot;en&quot;` in docs build ([#939](https://github.com/rapidsai/dask-cuda/pull/939)) [@galipremsagar](https://github.com/galipremsagar)

## 🚀 New Features

- Add communications bandwidth to benchmarks ([#938](https://github.com/rapidsai/dask-cuda/pull/938)) [@pentschev](https://github.com/pentschev)

## 🛠️ Improvements

- Pin `dask` &amp; `distributed` for release ([#965](https://github.com/rapidsai/dask-cuda/pull/965)) [@galipremsagar](https://github.com/galipremsagar)
- Test memory_limit=None for CUDAWorker ([#946](https://github.com/rapidsai/dask-cuda/pull/946)) [@wence-](https://github.com/wence-)
- benchmarks: Record total number of workers in dataframe ([#945](https://github.com/rapidsai/dask-cuda/pull/945)) [@wence-](https://github.com/wence-)
- Benchmark refactoring: tidy data and multi-node capability via `--scheduler-file` ([#940](https://github.com/rapidsai/dask-cuda/pull/940)) [@wence-](https://github.com/wence-)
- Add util functions to simplify printing benchmarks results ([#937](https://github.com/rapidsai/dask-cuda/pull/937)) [@pentschev](https://github.com/pentschev)
- Add --multiprocessing-method option to benchmarks ([#933](https://github.com/rapidsai/dask-cuda/pull/933)) [@wence-](https://github.com/wence-)
- Remove click pinning ([#932](https://github.com/rapidsai/dask-cuda/pull/932)) [@charlesbluca](https://github.com/charlesbluca)
- Remove compiler variables ([#929](https://github.com/rapidsai/dask-cuda/pull/929)) [@ajschmidt8](https://github.com/ajschmidt8)
- Unpin `dask` &amp; `distributed` for development ([#927](https://github.com/rapidsai/dask-cuda/pull/927)) [@galipremsagar](https://github.com/galipremsagar)

# dask-cuda 22.06.00 (7 Jun 2022)

## 🚨 Breaking Changes

- Upgrade `numba` pinning to be in-line with rest of rapids ([#912](https://github.com/rapidsai/dask-cuda/pull/912)) [@galipremsagar](https://github.com/galipremsagar)

## 🐛 Bug Fixes

- Reduce `test_cudf_cluster_device_spill` test and speed it up ([#918](https://github.com/rapidsai/dask-cuda/pull/918)) [@pentschev](https://github.com/pentschev)
- Update ImportError tests with --pre-import ([#914](https://github.com/rapidsai/dask-cuda/pull/914)) [@pentschev](https://github.com/pentschev)
- Add xfail mark to `test_pre_import_not_found` ([#908](https://github.com/rapidsai/dask-cuda/pull/908)) [@pentschev](https://github.com/pentschev)
- Increase spill tests timeout to 30 seconds ([#901](https://github.com/rapidsai/dask-cuda/pull/901)) [@pentschev](https://github.com/pentschev)
- Fix errors related with `distributed.worker.memory.terminate` ([#900](https://github.com/rapidsai/dask-cuda/pull/900)) [@pentschev](https://github.com/pentschev)
- Skip tests on import error for some optional packages ([#899](https://github.com/rapidsai/dask-cuda/pull/899)) [@pentschev](https://github.com/pentschev)
- Update auto host_memory computation when threads per worker &gt; 1 ([#896](https://github.com/rapidsai/dask-cuda/pull/896)) [@ayushdg](https://github.com/ayushdg)
- Update black to 22.3.0 ([#889](https://github.com/rapidsai/dask-cuda/pull/889)) [@charlesbluca](https://github.com/charlesbluca)
- Remove legacy `check_python_3` ([#886](https://github.com/rapidsai/dask-cuda/pull/886)) [@pentschev](https://github.com/pentschev)

## 📖 Documentation

- Add documentation for `RAPIDS_NO_INITIALIZE` ([#898](https://github.com/rapidsai/dask-cuda/pull/898)) [@charlesbluca](https://github.com/charlesbluca)
- Use upstream warning functions for CUDA initialization ([#894](https://github.com/rapidsai/dask-cuda/pull/894)) [@charlesbluca](https://github.com/charlesbluca)

## 🛠️ Improvements

- Pin `dask` and `distributed` for release ([#922](https://github.com/rapidsai/dask-cuda/pull/922)) [@galipremsagar](https://github.com/galipremsagar)
- Pin `dask` &amp; `distributed` for release ([#916](https://github.com/rapidsai/dask-cuda/pull/916)) [@galipremsagar](https://github.com/galipremsagar)
- Upgrade `numba` pinning to be in-line with rest of rapids ([#912](https://github.com/rapidsai/dask-cuda/pull/912)) [@galipremsagar](https://github.com/galipremsagar)
- Removing test of `cudf.merge_sorted()` ([#905](https://github.com/rapidsai/dask-cuda/pull/905)) [@madsbk](https://github.com/madsbk)
- Disable `include-ignored` coverage warnings ([#903](https://github.com/rapidsai/dask-cuda/pull/903)) [@pentschev](https://github.com/pentschev)
- Fix ci/local script ([#902](https://github.com/rapidsai/dask-cuda/pull/902)) [@Ethyling](https://github.com/Ethyling)
- Use conda to build python packages during GPU tests ([#897](https://github.com/rapidsai/dask-cuda/pull/897)) [@Ethyling](https://github.com/Ethyling)
- Pull `requirements.txt` into Conda recipe ([#893](https://github.com/rapidsai/dask-cuda/pull/893)) [@jakirkham](https://github.com/jakirkham)
- Unpin `dask` &amp; `distributed` for development ([#892](https://github.com/rapidsai/dask-cuda/pull/892)) [@galipremsagar](https://github.com/galipremsagar)
- Build packages using mambabuild ([#846](https://github.com/rapidsai/dask-cuda/pull/846)) [@Ethyling](https://github.com/Ethyling)

# dask-cuda 22.04.00 (6 Apr 2022)

## 🚨 Breaking Changes

- Introduce incompatible-types and enables spilling of CuPy arrays ([#856](https://github.com/rapidsai/dask-cuda/pull/856)) [@madsbk](https://github.com/madsbk)

## 🐛 Bug Fixes

- Resolve build issues / consistency with conda-forge packages ([#883](https://github.com/rapidsai/dask-cuda/pull/883)) [@charlesbluca](https://github.com/charlesbluca)
- Increase test_worker_force_spill_to_disk timeout ([#857](https://github.com/rapidsai/dask-cuda/pull/857)) [@pentschev](https://github.com/pentschev)

## 📖 Documentation

- Remove description from non-existing `--nprocs` CLI argument ([#852](https://github.com/rapidsai/dask-cuda/pull/852)) [@pentschev](https://github.com/pentschev)

## 🚀 New Features

- Add --pre-import/pre_import argument ([#854](https://github.com/rapidsai/dask-cuda/pull/854)) [@pentschev](https://github.com/pentschev)
- Remove support for UCX &lt; 1.11.1 ([#830](https://github.com/rapidsai/dask-cuda/pull/830)) [@pentschev](https://github.com/pentschev)

## 🛠️ Improvements

- Raise `ImportError` when platform is not Linux ([#885](https://github.com/rapidsai/dask-cuda/pull/885)) [@pentschev](https://github.com/pentschev)
- Temporarily disable new `ops-bot` functionality ([#880](https://github.com/rapidsai/dask-cuda/pull/880)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pin `dask` &amp; `distributed` ([#878](https://github.com/rapidsai/dask-cuda/pull/878)) [@galipremsagar](https://github.com/galipremsagar)
- Upgrade min `dask` &amp; `distributed` versions ([#872](https://github.com/rapidsai/dask-cuda/pull/872)) [@galipremsagar](https://github.com/galipremsagar)
- Add `.github/ops-bot.yaml` config file ([#871](https://github.com/rapidsai/dask-cuda/pull/871)) [@ajschmidt8](https://github.com/ajschmidt8)
- Make Dask CUDA work with the new WorkerMemoryManager abstraction ([#870](https://github.com/rapidsai/dask-cuda/pull/870)) [@shwina](https://github.com/shwina)
- Implement ProxifyHostFile.evict() ([#862](https://github.com/rapidsai/dask-cuda/pull/862)) [@madsbk](https://github.com/madsbk)
- Introduce incompatible-types and enables spilling of CuPy arrays ([#856](https://github.com/rapidsai/dask-cuda/pull/856)) [@madsbk](https://github.com/madsbk)
- Spill to disk clean up ([#853](https://github.com/rapidsai/dask-cuda/pull/853)) [@madsbk](https://github.com/madsbk)
- ProxyObject to support matrix multiplication ([#849](https://github.com/rapidsai/dask-cuda/pull/849)) [@madsbk](https://github.com/madsbk)
- Unpin max dask and distributed ([#847](https://github.com/rapidsai/dask-cuda/pull/847)) [@galipremsagar](https://github.com/galipremsagar)
- test_gds: skip if GDS is not available ([#845](https://github.com/rapidsai/dask-cuda/pull/845)) [@madsbk](https://github.com/madsbk)
- ProxyObject implement __array_function__ ([#843](https://github.com/rapidsai/dask-cuda/pull/843)) [@madsbk](https://github.com/madsbk)
- Add option to track RMM allocations ([#842](https://github.com/rapidsai/dask-cuda/pull/842)) [@shwina](https://github.com/shwina)

# dask-cuda 22.02.00 (2 Feb 2022)

## 🐛 Bug Fixes

- Ignore `DeprecationWarning` from `distutils.Version` classes ([#823](https://github.com/rapidsai/dask-cuda/pull/823)) [@pentschev](https://github.com/pentschev)
- Handle explicitly disabled UCX transports ([#820](https://github.com/rapidsai/dask-cuda/pull/820)) [@pentschev](https://github.com/pentschev)
- Fix regex pattern to match to in test_on_demand_debug_info ([#819](https://github.com/rapidsai/dask-cuda/pull/819)) [@pentschev](https://github.com/pentschev)
- Fix skipping GDS test if cucim is not installed ([#813](https://github.com/rapidsai/dask-cuda/pull/813)) [@pentschev](https://github.com/pentschev)
- Unpin Dask and Distributed versions ([#810](https://github.com/rapidsai/dask-cuda/pull/810)) [@pentschev](https://github.com/pentschev)
- Update to UCX-Py 0.24 ([#805](https://github.com/rapidsai/dask-cuda/pull/805)) [@pentschev](https://github.com/pentschev)

## 📖 Documentation

- Fix Dask-CUDA version to 22.02 ([#835](https://github.com/rapidsai/dask-cuda/pull/835)) [@jakirkham](https://github.com/jakirkham)
- Merge branch-21.12 into branch-22.02 ([#829](https://github.com/rapidsai/dask-cuda/pull/829)) [@pentschev](https://github.com/pentschev)
- Clarify `LocalCUDACluster`&#39;s `n_workers` docstrings ([#812](https://github.com/rapidsai/dask-cuda/pull/812)) [@pentschev](https://github.com/pentschev)

## 🚀 New Features

- Pin `dask` &amp; `distributed` versions ([#832](https://github.com/rapidsai/dask-cuda/pull/832)) [@galipremsagar](https://github.com/galipremsagar)
- Expose rmm-maximum_pool_size argument ([#827](https://github.com/rapidsai/dask-cuda/pull/827)) [@VibhuJawa](https://github.com/VibhuJawa)
- Simplify UCX configs, permitting UCX_TLS=all ([#792](https://github.com/rapidsai/dask-cuda/pull/792)) [@pentschev](https://github.com/pentschev)

## 🛠️ Improvements

- Add avg and std calculation for time and throughput ([#828](https://github.com/rapidsai/dask-cuda/pull/828)) [@quasiben](https://github.com/quasiben)
- sizeof test: increase tolerance ([#825](https://github.com/rapidsai/dask-cuda/pull/825)) [@madsbk](https://github.com/madsbk)
- Query UCX-Py from gpuCI versioning service ([#818](https://github.com/rapidsai/dask-cuda/pull/818)) [@pentschev](https://github.com/pentschev)
- Standardize Distributed config separator in get_ucx_config ([#806](https://github.com/rapidsai/dask-cuda/pull/806)) [@pentschev](https://github.com/pentschev)
- Fixed `ProxyObject.__del__` to use the new Disk IO API from #791 ([#802](https://github.com/rapidsai/dask-cuda/pull/802)) [@madsbk](https://github.com/madsbk)
- GPUDirect Storage (GDS) support for spilling ([#793](https://github.com/rapidsai/dask-cuda/pull/793)) [@madsbk](https://github.com/madsbk)
- Disk IO interface ([#791](https://github.com/rapidsai/dask-cuda/pull/791)) [@madsbk](https://github.com/madsbk)

# dask-cuda 21.12.00 (9 Dec 2021)

## 🐛 Bug Fixes

- Remove automatic `doc` labeler ([#807](https://github.com/rapidsai/dask-cuda/pull/807)) [@pentschev](https://github.com/pentschev)
- Add create_cuda_context UCX config from Distributed ([#801](https://github.com/rapidsai/dask-cuda/pull/801)) [@pentschev](https://github.com/pentschev)
- Ignore deprecation warnings from pkg_resources ([#784](https://github.com/rapidsai/dask-cuda/pull/784)) [@pentschev](https://github.com/pentschev)
- Fix parsing of device by UUID ([#780](https://github.com/rapidsai/dask-cuda/pull/780)) [@pentschev](https://github.com/pentschev)
- Avoid creating CUDA context in LocalCUDACluster parent process ([#765](https://github.com/rapidsai/dask-cuda/pull/765)) [@pentschev](https://github.com/pentschev)
- Remove gen_cluster spill tests ([#758](https://github.com/rapidsai/dask-cuda/pull/758)) [@pentschev](https://github.com/pentschev)
- Update memory_pause_fraction in test_spill ([#757](https://github.com/rapidsai/dask-cuda/pull/757)) [@pentschev](https://github.com/pentschev)

## 📖 Documentation

- Add troubleshooting page with PCI Bus ID issue description ([#777](https://github.com/rapidsai/dask-cuda/pull/777)) [@pentschev](https://github.com/pentschev)

## 🚀 New Features

- Handle UCX-Py FutureWarning on UCX &lt; 1.11.1 deprecation ([#799](https://github.com/rapidsai/dask-cuda/pull/799)) [@pentschev](https://github.com/pentschev)
- Pin max `dask` &amp; `distributed` versions ([#794](https://github.com/rapidsai/dask-cuda/pull/794)) [@galipremsagar](https://github.com/galipremsagar)
- Update to UCX-Py 0.23 ([#752](https://github.com/rapidsai/dask-cuda/pull/752)) [@pentschev](https://github.com/pentschev)

## 🛠️ Improvements

- Fix spill-to-disk triggered by Dask explicitly ([#800](https://github.com/rapidsai/dask-cuda/pull/800)) [@madsbk](https://github.com/madsbk)
- Fix Changelog Merge Conflicts for `branch-21.12` ([#797](https://github.com/rapidsai/dask-cuda/pull/797)) [@ajschmidt8](https://github.com/ajschmidt8)
- Use unittest.mock.patch for all os.environ tests ([#787](https://github.com/rapidsai/dask-cuda/pull/787)) [@pentschev](https://github.com/pentschev)
- Logging when RMM allocation fails ([#782](https://github.com/rapidsai/dask-cuda/pull/782)) [@madsbk](https://github.com/madsbk)
- Tally IDs instead of device buffers directly ([#779](https://github.com/rapidsai/dask-cuda/pull/779)) [@madsbk](https://github.com/madsbk)
- Avoid proxy object aliasing ([#775](https://github.com/rapidsai/dask-cuda/pull/775)) [@madsbk](https://github.com/madsbk)
- Test of sizeof proxy object ([#774](https://github.com/rapidsai/dask-cuda/pull/774)) [@madsbk](https://github.com/madsbk)
- gc.collect when spilling on demand ([#771](https://github.com/rapidsai/dask-cuda/pull/771)) [@madsbk](https://github.com/madsbk)
- Reenable explicit comms tests ([#770](https://github.com/rapidsai/dask-cuda/pull/770)) [@madsbk](https://github.com/madsbk)
- Simplify JIT-unspill and writing docs ([#768](https://github.com/rapidsai/dask-cuda/pull/768)) [@madsbk](https://github.com/madsbk)
- Increase CUDAWorker close timeout ([#764](https://github.com/rapidsai/dask-cuda/pull/764)) [@pentschev](https://github.com/pentschev)
- Ignore known but expected test warnings ([#759](https://github.com/rapidsai/dask-cuda/pull/759)) [@pentschev](https://github.com/pentschev)
- Spilling on demand ([#756](https://github.com/rapidsai/dask-cuda/pull/756)) [@madsbk](https://github.com/madsbk)
- Revert &quot;Temporarily skipping some tests because of a bug in Dask ([#753)&quot; (#754](https://github.com/rapidsai/dask-cuda/pull/753)&quot; (#754)) [@madsbk](https://github.com/madsbk)
- Temporarily skipping some tests because of a bug in Dask ([#753](https://github.com/rapidsai/dask-cuda/pull/753)) [@madsbk](https://github.com/madsbk)
- Removing the `FrameProxyObject` workaround ([#751](https://github.com/rapidsai/dask-cuda/pull/751)) [@madsbk](https://github.com/madsbk)
- Use cuDF Frame instead of Table ([#748](https://github.com/rapidsai/dask-cuda/pull/748)) [@madsbk](https://github.com/madsbk)
- Remove proxy object locks ([#747](https://github.com/rapidsai/dask-cuda/pull/747)) [@madsbk](https://github.com/madsbk)
- Unpin `dask` &amp; `distributed` in CI ([#742](https://github.com/rapidsai/dask-cuda/pull/742)) [@galipremsagar](https://github.com/galipremsagar)
- Update SSHCluster usage in benchmarks with new CUDAWorker ([#326](https://github.com/rapidsai/dask-cuda/pull/326)) [@pentschev](https://github.com/pentschev)

# dask-cuda 21.10.00 (7 Oct 2021)

## 🐛 Bug Fixes

- Drop test setting UCX global options via Dask config ([#738](https://github.com/rapidsai/dask-cuda/pull/738)) [@pentschev](https://github.com/pentschev)
- Prevent CUDA context errors when testing on single-GPU ([#737](https://github.com/rapidsai/dask-cuda/pull/737)) [@pentschev](https://github.com/pentschev)
- Handle `ucp` import error during `initialize()` ([#729](https://github.com/rapidsai/dask-cuda/pull/729)) [@pentschev](https://github.com/pentschev)
- Check if CUDA context was created in distributed.comm.ucx ([#722](https://github.com/rapidsai/dask-cuda/pull/722)) [@pentschev](https://github.com/pentschev)
- Fix registering correct dispatches for `cudf.Index` ([#718](https://github.com/rapidsai/dask-cuda/pull/718)) [@galipremsagar](https://github.com/galipremsagar)
- Register `percentile_lookup` for `FrameProxyObject` ([#716](https://github.com/rapidsai/dask-cuda/pull/716)) [@galipremsagar](https://github.com/galipremsagar)
- Leave interface unset when ucx_net_devices unset in LocalCUDACluster ([#711](https://github.com/rapidsai/dask-cuda/pull/711)) [@pentschev](https://github.com/pentschev)
- Update to UCX-Py 0.22 ([#710](https://github.com/rapidsai/dask-cuda/pull/710)) [@pentschev](https://github.com/pentschev)
- Missing fixes to Distributed config namespace refactoring ([#703](https://github.com/rapidsai/dask-cuda/pull/703)) [@pentschev](https://github.com/pentschev)
- Reset UCX-Py after rdmacm tests run ([#702](https://github.com/rapidsai/dask-cuda/pull/702)) [@pentschev](https://github.com/pentschev)
- Skip DGX InfiniBand tests when &quot;rc&quot; transport is unavailable ([#701](https://github.com/rapidsai/dask-cuda/pull/701)) [@pentschev](https://github.com/pentschev)
- Update UCX config namespace ([#695](https://github.com/rapidsai/dask-cuda/pull/695)) [@pentschev](https://github.com/pentschev)
- Bump isort hook version ([#682](https://github.com/rapidsai/dask-cuda/pull/682)) [@charlesbluca](https://github.com/charlesbluca)

## 📖 Documentation

- Update more docs for UCX 1.11+ ([#720](https://github.com/rapidsai/dask-cuda/pull/720)) [@pentschev](https://github.com/pentschev)
- Forward-merge branch-21.08 to branch-21.10 ([#707](https://github.com/rapidsai/dask-cuda/pull/707)) [@jakirkham](https://github.com/jakirkham)

## 🚀 New Features

- Warn if CUDA context is created on incorrect device with `LocalCUDACluster` ([#719](https://github.com/rapidsai/dask-cuda/pull/719)) [@pentschev](https://github.com/pentschev)
- Add `--benchmark-json` option to all benchmarks ([#700](https://github.com/rapidsai/dask-cuda/pull/700)) [@charlesbluca](https://github.com/charlesbluca)
- Remove Distributed tests from CI ([#699](https://github.com/rapidsai/dask-cuda/pull/699)) [@pentschev](https://github.com/pentschev)
- Add device memory limit argument to benchmarks ([#683](https://github.com/rapidsai/dask-cuda/pull/683)) [@charlesbluca](https://github.com/charlesbluca)
- Support for LocalCUDACluster with MIG ([#674](https://github.com/rapidsai/dask-cuda/pull/674)) [@akaanirban](https://github.com/akaanirban)

## 🛠️ Improvements

- Pin max `dask` and `distributed` versions to `2021.09.1` ([#735](https://github.com/rapidsai/dask-cuda/pull/735)) [@galipremsagar](https://github.com/galipremsagar)
- Implements a ProxyManagerDummy for convenience ([#733](https://github.com/rapidsai/dask-cuda/pull/733)) [@madsbk](https://github.com/madsbk)
- Add `__array_ufunc__` support for `ProxyObject` ([#731](https://github.com/rapidsai/dask-cuda/pull/731)) [@galipremsagar](https://github.com/galipremsagar)
- Use `has_cuda_context` from Distributed ([#723](https://github.com/rapidsai/dask-cuda/pull/723)) [@pentschev](https://github.com/pentschev)
- Fix deadlock and simplify proxy tracking ([#712](https://github.com/rapidsai/dask-cuda/pull/712)) [@madsbk](https://github.com/madsbk)
- JIT-unspill: support spilling to/from disk ([#708](https://github.com/rapidsai/dask-cuda/pull/708)) [@madsbk](https://github.com/madsbk)
- Tests: replacing the obsolete cudf.testing._utils.assert_eq calls ([#706](https://github.com/rapidsai/dask-cuda/pull/706)) [@madsbk](https://github.com/madsbk)
- JIT-unspill: warn when spill to disk triggers ([#705](https://github.com/rapidsai/dask-cuda/pull/705)) [@madsbk](https://github.com/madsbk)
- Remove max version pin for `dask` &amp; `distributed` on development branch ([#693](https://github.com/rapidsai/dask-cuda/pull/693)) [@galipremsagar](https://github.com/galipremsagar)
- ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#675](https://github.com/rapidsai/dask-cuda/pull/675)) [@dillon-cullinan](https://github.com/dillon-cullinan)

# dask-cuda 21.08.00 (4 Aug 2021)

## 🐛 Bug Fixes

- Use aliases to check for installed UCX version ([#692](https://github.com/rapidsai/dask-cuda/pull/692)) [@pentschev](https://github.com/pentschev)
- Don&#39;t install Dask main branch in CI for 21.08 release ([#687](https://github.com/rapidsai/dask-cuda/pull/687)) [@pentschev](https://github.com/pentschev)
- Skip test_get_ucx_net_devices_raises on UCX &gt;= 1.11.0 ([#685](https://github.com/rapidsai/dask-cuda/pull/685)) [@pentschev](https://github.com/pentschev)
- Fix NVML index usage in CUDAWorker/LocalCUDACluster ([#671](https://github.com/rapidsai/dask-cuda/pull/671)) [@pentschev](https://github.com/pentschev)
- Add --protocol flag to dask-cuda-worker ([#670](https://github.com/rapidsai/dask-cuda/pull/670)) [@jacobtomlinson](https://github.com/jacobtomlinson)
- Fix `assert_eq` related imports ([#663](https://github.com/rapidsai/dask-cuda/pull/663)) [@galipremsagar](https://github.com/galipremsagar)
- Small tweaks to make compatible with dask-mpi ([#656](https://github.com/rapidsai/dask-cuda/pull/656)) [@jacobtomlinson](https://github.com/jacobtomlinson)
- Remove Dask version pin ([#647](https://github.com/rapidsai/dask-cuda/pull/647)) [@pentschev](https://github.com/pentschev)
- Fix CUDA_VISIBLE_DEVICES tests ([#638](https://github.com/rapidsai/dask-cuda/pull/638)) [@pentschev](https://github.com/pentschev)
- Add `make_meta_dispatch` handling ([#637](https://github.com/rapidsai/dask-cuda/pull/637)) [@galipremsagar](https://github.com/galipremsagar)
- Update UCX-Py version in CI to 0.21.* ([#636](https://github.com/rapidsai/dask-cuda/pull/636)) [@pentschev](https://github.com/pentschev)

## 📖 Documentation

- Deprecation warning for ucx_net_devices=&#39;auto&#39; on UCX 1.11+ ([#681](https://github.com/rapidsai/dask-cuda/pull/681)) [@pentschev](https://github.com/pentschev)
- Update documentation on InfiniBand with UCX &gt;= 1.11 ([#669](https://github.com/rapidsai/dask-cuda/pull/669)) [@pentschev](https://github.com/pentschev)
- Merge branch-21.06 ([#622](https://github.com/rapidsai/dask-cuda/pull/622)) [@pentschev](https://github.com/pentschev)

## 🚀 New Features

- Treat Deprecation/Future warnings as errors ([#672](https://github.com/rapidsai/dask-cuda/pull/672)) [@pentschev](https://github.com/pentschev)
- Update parse_bytes imports to resolve deprecation warnings ([#662](https://github.com/rapidsai/dask-cuda/pull/662)) [@pentschev](https://github.com/pentschev)

## 🛠️ Improvements

- Pin max `dask` &amp; `distributed` versions ([#686](https://github.com/rapidsai/dask-cuda/pull/686)) [@galipremsagar](https://github.com/galipremsagar)
- Fix DGX tests warnings on RMM pool size and file not closed ([#673](https://github.com/rapidsai/dask-cuda/pull/673)) [@pentschev](https://github.com/pentschev)
- Remove dot calling style for pytest ([#661](https://github.com/rapidsai/dask-cuda/pull/661)) [@quasiben](https://github.com/quasiben)
- get_device_memory_objects(): dispatch on cudf.core.frame.Frame ([#658](https://github.com/rapidsai/dask-cuda/pull/658)) [@madsbk](https://github.com/madsbk)
- Fix `21.08` forward-merge conflicts ([#655](https://github.com/rapidsai/dask-cuda/pull/655)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix conflicts in `643` ([#644](https://github.com/rapidsai/dask-cuda/pull/644)) [@ajschmidt8](https://github.com/ajschmidt8)

# dask-cuda 21.06.00 (9 Jun 2021)

## 🐛 Bug Fixes

- Handle `import`ing relocated dispatch functions ([#623](https://github.com/rapidsai/dask-cuda/pull/623)) [@jakirkham](https://github.com/jakirkham)
- Fix DGX tests for UCX 1.9 ([#619](https://github.com/rapidsai/dask-cuda/pull/619)) [@pentschev](https://github.com/pentschev)
- Add PROJECTS var ([#614](https://github.com/rapidsai/dask-cuda/pull/614)) [@ajschmidt8](https://github.com/ajschmidt8)

## 📖 Documentation

- Bump docs copyright year ([#616](https://github.com/rapidsai/dask-cuda/pull/616)) [@charlesbluca](https://github.com/charlesbluca)
- Update RTD site to redirect to RAPIDS docs ([#615](https://github.com/rapidsai/dask-cuda/pull/615)) [@charlesbluca](https://github.com/charlesbluca)
- Document DASK_JIT_UNSPILL ([#604](https://github.com/rapidsai/dask-cuda/pull/604)) [@madsbk](https://github.com/madsbk)

## 🚀 New Features

- Disable reuse endpoints with UCX &gt;= 1.11 ([#620](https://github.com/rapidsai/dask-cuda/pull/620)) [@pentschev](https://github.com/pentschev)

## 🛠️ Improvements

- Adding profiling to dask shuffle ([#625](https://github.com/rapidsai/dask-cuda/pull/625)) [@arunraman](https://github.com/arunraman)
- Update `CHANGELOG.md` links for calver ([#618](https://github.com/rapidsai/dask-cuda/pull/618)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fixing Dataframe merge benchmark ([#617](https://github.com/rapidsai/dask-cuda/pull/617)) [@madsbk](https://github.com/madsbk)
- Fix DGX tests for UCX 1.10+ ([#613](https://github.com/rapidsai/dask-cuda/pull/613)) [@pentschev](https://github.com/pentschev)
- Update docs build script ([#612](https://github.com/rapidsai/dask-cuda/pull/612)) [@ajschmidt8](https://github.com/ajschmidt8)

# dask-cuda 0.19.0 (21 Apr 2021)

## 🐛 Bug Fixes

- Pin Dask and Distributed &lt;=2021.04.0 ([#585](https://github.com/rapidsai/dask-cuda/pull/585)) [@pentschev](https://github.com/pentschev)
- Unblock CI by xfailing test_dataframe_merge_empty_partitions ([#581](https://github.com/rapidsai/dask-cuda/pull/581)) [@pentschev](https://github.com/pentschev)
- Install Dask + Distributed from `main` ([#546](https://github.com/rapidsai/dask-cuda/pull/546)) [@jakirkham](https://github.com/jakirkham)
- Replace compute() calls on CuPy benchmarks by persist() ([#537](https://github.com/rapidsai/dask-cuda/pull/537)) [@pentschev](https://github.com/pentschev)

## 📖 Documentation

- Add standalone examples of UCX usage ([#551](https://github.com/rapidsai/dask-cuda/pull/551)) [@charlesbluca](https://github.com/charlesbluca)
- Improve UCX documentation and examples ([#545](https://github.com/rapidsai/dask-cuda/pull/545)) [@charlesbluca](https://github.com/charlesbluca)
- Auto-merge branch-0.18 to branch-0.19 ([#538](https://github.com/rapidsai/dask-cuda/pull/538)) [@GPUtester](https://github.com/GPUtester)

## 🚀 New Features

- Add option to enable RMM logging ([#542](https://github.com/rapidsai/dask-cuda/pull/542)) [@charlesbluca](https://github.com/charlesbluca)
- Add capability to log spilling ([#442](https://github.com/rapidsai/dask-cuda/pull/442)) [@pentschev](https://github.com/pentschev)

## 🛠️ Improvements

- Fix UCX examples for InfiniBand ([#556](https://github.com/rapidsai/dask-cuda/pull/556)) [@charlesbluca](https://github.com/charlesbluca)
- Fix list to tuple conversion ([#555](https://github.com/rapidsai/dask-cuda/pull/555)) [@madsbk](https://github.com/madsbk)
- Add column masking operation for CuPy benchmarking ([#553](https://github.com/rapidsai/dask-cuda/pull/553)) [@jakirkham](https://github.com/jakirkham)
- Update Changelog Link ([#550](https://github.com/rapidsai/dask-cuda/pull/550)) [@ajschmidt8](https://github.com/ajschmidt8)
- cuDF-style operations &amp; NVTX annotations for local CuPy benchmark ([#548](https://github.com/rapidsai/dask-cuda/pull/548)) [@charlesbluca](https://github.com/charlesbluca)
- Prepare Changelog for Automation ([#543](https://github.com/rapidsai/dask-cuda/pull/543)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add --enable-rdmacm flag to benchmarks utils ([#539](https://github.com/rapidsai/dask-cuda/pull/539)) [@pentschev](https://github.com/pentschev)
- ProxifyHostFile: tracking of external objects ([#527](https://github.com/rapidsai/dask-cuda/pull/527)) [@madsbk](https://github.com/madsbk)
- Test broadcast merge in local_cudf_merge benchmark ([#507](https://github.com/rapidsai/dask-cuda/pull/507)) [@rjzamora](https://github.com/rjzamora)

# dask-cuda 0.18.0 (24 Feb 2021)

## Breaking Changes 🚨

- Explicit-comms house cleaning (#515) @madsbk

## Bug Fixes 🐛

- Fix device synchronization in local_cupy benchmark (#518) @pentschev
- Proxify register lazy (#492) @madsbk
- Work on deadlock issue 431 (#490) @madsbk
- Fix usage of --dashboard-address in dask-cuda-worker (#487) @pentschev
- Fail if scheduler starts with &#39;-&#39; in dask-cuda-worker (#485) @pentschev

## Documentation 📖

- Add device synchonization for local CuPy benchmarks with Dask profiling (#533) @charlesbluca

## New Features 🚀

- Shuffle benchmark (#496) @madsbk

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#531) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#528) @Ethyling
- Add operations/arguments to local CuPy array benchmark (#524) @charlesbluca
- Explicit-comms house cleaning (#515) @madsbk
- Fixing fixed-attribute-proxy-object-test (#511) @madsbk
- Prepare Changelog for Automation (#509) @ajschmidt8
- remove conditional check to start conda uploads (#504) @jolorunyomi
- ProxyObject: ignore initial fixed attribute errors (#503) @madsbk
- JIT-unspill: fix potential deadlock (#501) @madsbk
- Hostfile: register the removal of an existing key (#500) @madsbk
- proxy_object: cleanup type dispatching (#497) @madsbk
- Redesign and implementation of dataframe shuffle (#494) @madsbk
- Add --threads-per-worker option to benchmarks (#489) @pentschev
- Extend CuPy benchmark with more operations (#488) @pentschev
- Auto-label PRs based on their content (#480) @jolorunyomi
- CI: cleanup style check (#477) @madsbk
- Individual CUDA object spilling (#451) @madsbk
- FIX Move codecov upload to gpu build script (#450) @dillon-cullinan
- Add support for connecting a CUDAWorker to a cluster object (#428) @jacobtomlinson

# 0.17.0

- Fix benchmark output when scheduler address is specified (#414) @quasiben
- Fix typo in benchmark utils (#416) @quasiben
- More RMM options in benchmarks (#419) @quasiben
- Add utility function to establish all-to-all connectivity upon request (#420) @quasiben
- Filter `rmm_pool_size` warnings in benchmarks (#422) @pentschev
- Add functionality to plot cuDF benchmarks (#423) @quasiben
- Decrease data size to shorten spilling tests time (#422) @pentschev
- Temporarily xfail explicit-comms tests (#432) @pentschev
- Add codecov.yml and ignore uncovered files (#433) @pentschev
- Do not skip DGX/TCP tests when ucp is not installed (#436) @pentschev
- Support UUID in CUDA_VISIBLE_DEVICES (#437) @pentschev
- Unify `device_memory_limit` parsing and set default to 0.8 (#439) @pentschev
- Update and clean gpuCI scripts (#440) @msadang
- Add notes on controlling number of workers to docs (#441) @quasiben
- Add CPU support to CuPy transpose sum benchmark (#444) @pentschev
- Update builddocs dependency requirements (#447) @quasiben
- Fix versioneer (#448) @jakirkham
- Cleanup conda recipe (#449) @jakirkham
- Fix `pip install` issues with new resolver (#454) @jakirkham
- Make threads per worker consistent (#456) @pentschev
- Support for ProxyObject binary operations (#458) @madsbk
- Support for ProxyObject pickling (#459) @madsbk
- Clarify RMM pool is a per-worker attribute on docs (#462) @pentschev
- Fix typo on specializations docs (#463) @vfdev-5

# 0.16.0

- Parse pool size only when set (#396) @quasiben
- Improve CUDAWorker scheduler-address parsing and __init__ (#397) @necaris
- Add benchmark for `da.map_overlap` (#399) @jakirkham
- Explicit-comms: dataframe shuffle (#401) @madsbk
- Use new NVTX module (#406) @pentschev
- Run Dask's NVML tests (#408) @quasiben
- Skip tests that require cuDF/UCX-Py, when not installed (#411) @pentschev

# 0.15.0

- Fix-up versioneer (#305) @jakirkham
- Require Distributed 2.15.0+ (#306) @jakirkham
- Rely on Dask's ability to serialize collections (#307) @jakirkham
- Ensure CI installs GPU build of UCX (#308) @pentschev
- Skip 2nd serialization pass of `DeviceSerialized` (#309) @jakirkham
- Fix tests related to latest RMM changes (#310) @pentschev
- Fix dask-cuda-worker's interface argument (#314) @pentschev
- Check only for memory type during test_get_device_total_memory (#315) @pentschev
- Fix and improve DGX tests (#316) @pentschev
- Install dependencies via meta package (#317) @raydouglass
- Fix errors when TLS files are not specified (#320) @pentschev
- Refactor dask-cuda-worker into CUDAWorker class (#324) @jacobtomlinson
- Add missing __init__.py to dask_cuda/cli (#327) @pentschev
- Add Dask distributed GPU tests to CI (#329) @quasiben
- Fix rmm_pool_size argument name in docstrings (#329) @quasiben
- Add CPU support to benchmarks (#338) @quasiben
- Fix isort configuration (#339) @madsbk
- Explicit-comms: cleanup and bug fix (#340) @madsbk
- Add support for RMM managed memory (#343) @pentschev
- Update docker image in local build script (#345) @sean-frye
- Support pickle protocol 5 based spilling (#349) @jakirkham
- Use get_n_gpus for RMM test with dask-cuda-worker (#356) @pentschev
- Update RMM tests based on deprecated CNMeM (#359) @jakirkham
- Fix a black error in explicit comms (#360) @jakirkham
- Fix an `isort` error (#360) @jakirkham
- Set `RMM_NO_INITIALIZE` environment variable (#363) @quasiben
- Fix bash lines in docs (#369) @quasiben
- Replace `RMM_NO_INITIALIZE` with `RAPIDS_NO_INITIALIZE` (#371) @jakirkham
- Fixes for docs and RTD updates (#373) @quasiben
- Confirm DGX tests are running baremetal (#376) @pentschev
- Set RAPIDS_NO_INITIALIZE at the top of CUDAWorker/LocalCUDACluster (#379) @pentschev
- Change pytest's basetemp in CI build script (#380) @pentschev
- Pin Numba version to exclude 0.51.0 (#385) @quasiben

# 0.14.0

- Publish branch-0.14 to conda (#262) @trxcllnt
- Fix behavior for `memory_limit=0` (#269) @pentschev
- Raise serialization errors when spilling (#272) @jakirkham
- Fix dask-cuda-worker memory_limit (#279) @pentschev
- Add NVTX annotations for spilling (#282) @pentschev
- Skip existing on conda uploads (#284) @raydouglass
- Local gpuCI build script (#285) @efajardo-nv
- Remove deprecated DGX class (#286) @pentschev
- Add RDMACM support (#287) @pentschev
- Read the Docs Setup (#290) @quasiben
- Raise ValueError when ucx_net_devices="auto" and IB is disabled (#291) @pentschev
- Multi-node benchmarks (#293) @pentschev
- Add docs for UCX (#294) @pentschev
- Add `--runs` argument to CuPy benchmark (#295) @pentschev
- Fixing LocalCUDACluster example. Adding README links to docs (#297) @randerzander
- Add `nfinal` argument to shuffle_group, required in Dask >= 2.17 (#299) @pentschev
- Initialize parent process' UCX configuration (#301) @pentschev
- Add Read the Docs link (#302) @jakirkham

# 0.13.0

- Use RMM's `DeviceBuffer` directly (#235) @jakirkham
- Add RMM pool support from dask-cuda-worker/LocalCUDACluster (#236) @pentschev
- Restrict CuPy to <7.2 (#239) @quasiben
- Fix UCX configurations (#246) @pentschev
- Respect `temporary-directory` config for spilling (#247) @jakirkham
- Relax CuPy pin (#248) @jakirkham
- Added `ignore_index` argument to `partition_by_hash()` (#253) @madsbk
- Use `"dask"` serialization to move to/from host (#256) @jakirkham
- Drop Numba `DeviceNDArray` code for `sizeof` (#257) @jakirkham
- Support spilling of device objects in dictionaries (#260) @madsbk

# 0.12.0

- Add ucx-py dependency to CI (#212) @raydouglass
- Follow-up revision of local_cudf_merge benchmark (#213) @rjzamora
- Add codeowners file (#217) @raydouglass
- Add pypi upload script (#218) @raydouglass
- Skip existing on PyPi uploads (#219) @raydouglass
- Make benchmarks use rmm_cupy_allocator (#220) @madsbk
- cudf-merge-benchmark now reports throughput (#222) @madsbk
- Fix dask-cuda-worker --interface/--net-devices docs (#223) @pentschev
- Use RMM for serialization when available (#227) @pentschev

# 0.11.0

- Use UCX-Py initialization API (#152) @pentschev
- Remove all CUDA labels (#160) @mike-wendt
- Setting UCX options through dask global config (#168) @madsbk
- Make test_cudf_device_spill xfail (#170) @pentschev
- Updated CI, cleanup tests and reformat Python files (#171) @madsbk
- Fix GPU dependency versions (#173) @dillon-cullinan
- Set LocalCUDACluster n_workers equal to the length of CUDA_VISIBLE_DEVICES (#174) @mrocklin
- Simplify dask-cuda code (#175) @madsbk
- DGX inherit from LocalCUDACluster (#177) @madsbk
- Single-node CUDA benchmarks (#179) @madsbk
- Set TCP for UCX tests (#180) @pentschev
- Single-node cuDF merge benchmarks (#183) @madsbk
- Add black and isort checks in CI (#185) @pentschev
- Remove outdated xfail/importorskip test entries (#188) @pentschev
- Use UCX-Py's TopologicalDistance to determine IB interfaces in DGX (#189) @pentschev
- Dask Performance Report (#192) @madsbk
- Allow test_cupy_device_spill to xfail (#195) @pentschev
- Use ucx-py from rapidsai-nightly in CI (#196) @pentschev
- LocalCUDACluster sets closest network device (#200) @madsbk
- Expand cudf-merge benchmark (#201) @rjzamora
- Added --runs to merge benchmark (#202) @madsbk
- Move UCX code to LocalCUDACluster and deprecate DGX (#205) @pentschev
- Add markdown output option to cuDF merge benchmark (#208) @quasiben

# 0.10.0

- Change the updated new_worker_spec API for upstream (#128) @mrocklin
- Update TOTAL_MEMORY to match new distributed MEMORY_LIMIT (#131) @pentschev
- Bum Dask requirement to 2.4 (#133) @mrocklin
- Use YYMMDD tag in nightly build (#134) @mluukkainen
- Automatically determine CPU affinity (#138) @pentschev
- Fix full memory use check testcase (#139) @ksangeek
- Use pynvml to get memory info without creating CUDA context (#140) @pentschev
- Pass missing local_directory to Nanny from dask-cuda-worker (#141) @pentschev
- New worker_spec function for worker recipes (#147) @pentschev
- Add new Scheduler class supporting environment variables (#149) @pentschev
- Support for TCP over UCX (#152) @pentschev

# 0.9.0

- Fix serialization of collections and bump dask to 2.3.0 (#118) @pentschev
- Add versioneer (#88) @matthieubulte
- Python CodeCov Integration (#91) @dillon-cullinan
- Update cudf, dask, dask-cudf, distributed version requirements (#97) @pentschev
- Improve device memory spilling performance (#98) @pentschev
- Update dask-cuda for dask 2.2 (#101) @mrocklin
- Streamline CUDA_REL environment variable (#102) @okoskinen
- Replace ncores= parameter with nthreads= (#101) @mrocklin
- Fix remove CodeCov upload from build script (#115) @dillon-cullinan
- Remove CodeCov upload (#116) @dillon-cullinan

# 0.8.0

-  Add device memory spill support (LRU-based only) (#51) @pentschev
-  Update CI dependency to CuPy 6.0.0 (#53) @pentschev
-  Add a hard-coded DGX configuration (#46) (#70) @mrocklin
-  Fix LocalCUDACluster data spilling and its test (#67) @pentschev
-  Add test skipping functionality to build.sh (#71) @dillon-cullinan
-  Replace use of ncores= keywords with nthreads= (#75) @mrocklin
-  Fix device memory spilling with cuDF (#65) @pentschev
-  LocalCUDACluster calls _correct_state() to ensure workers started (#78) @pentschev
-  Delay some of spilling test assertions (#80) @pentschev
