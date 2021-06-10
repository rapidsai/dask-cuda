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
