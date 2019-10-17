0.10
----

- Change the updated new_worker_spec API for upstream (#128) `Matthew Rocklin`_
- Update TOTAL_MEMORY to match new distributed MEMORY_LIMIT (#131) `Peter Andreas Entschev`_
- Bum Dask requirement to 2.4 (#133) `Matthew Rocklin`_
- Use YYMMDD tag in nightly build (#134) `Markku Luukkainen`_
- Automatically determine CPU affinity (#138) `Peter Andreas Entschev`_
- Fix full memory use check testcase (#139) `Sangeeth Keeriyadath`_
- Use pynvml to get memory info without creating CUDA context (#140) `Peter Andreas Entschev`_
- Pass missing local_directory to Nanny from dask-cuda-worker (#141) `Peter Andreas Entschev`_
- New worker_spec function for worker recipes (#147) `Peter Andreas Entschev`_
- Add new Scheduler class supporting environment variables (#149) `Peter Andreas Entschev`_
- Support for TCP over UCX (#152) `Peter Andreas Entschev`_


.. _`Matthew Rocklin`: https://github.com/mrocklin
.. _`Peter Andreas Entschev`: https://github.com/pentschev
.. _`Markku Luukkainen`: https://github.com/mluukkainen
.. _`Sangeeth Keeriyadath`: https://github.com/ksangeek

0.9
---

- Fix serialization of collections and bump dask to 2.3.0 (#118) `Peter Andreas Entschev`_
- Add versioneer (#88) `Matthieu Bulte`_
- Python CodeCov Integration (#91) `Dillon Cullinan`_
- Update cudf, dask, dask-cudf, distributed version requirements (#97) `Peter Andreas Entschev`_
- Improve device memory spilling performance (#98) `Peter Andreas Entschev`_
- Update dask-cuda for dask 2.2 (#101) `Matthew Rocklin`_
- Streamline CUDA_REL environment variable (#102) `Olli Koskinen`_
- Replace ncores= parameter with nthreads= (#101) `Matthew Rocklin`_
- Fix remove CodeCov upload from build script (#115) `Dillon Cullinan`_
- Remove CodeCov upload (#116) `Dillon Cullinan`_

.. _`Matthieu Bulte`: https://github.com/matthieubulte
.. _`Dillon Cullinan`: https://github.com/dillon-cullinan
.. _`Peter Andreas Entschev`: https://github.com/pentschev
.. _`Matthew Rocklin`: https://github.com/mrocklin
.. _`Olli Koskinen`: https://github.com/okoskinen

0.8
---

-  Add device memory spill support (LRU-based only) (#51) `Peter Andreas Entschev`_
-  Update CI dependency to CuPy 6.0.0 (#53) `Peter Andreas Entschev`_
-  Add a hard-coded DGX configuration (#46) (#70) `Matthew Rocklin`_
-  Fix LocalCUDACluster data spilling and its test (#67) `Peter Andreas Entschev`_
-  Add test skipping functionality to build.sh (#71) `Dillon Cullinan`_
-  Replace use of ncores= keywords with nthreads= (#75) `Matthew Rocklin`_
-  Fix device memory spilling with cuDF (#65) `Peter Andreas Entschev`_
-  LocalCUDACluster calls _correct_state() to ensure workers started (#78) `Peter Andreas Entschev`_
-  Delay some of spilling test assertions (#80) `Peter Andreas Entschev`_


.. _`Peter Andreas Entschev`: https://github.com/pentschev
.. _`Matthew Rocklin`: https://github.com/mrocklin
.. _`Dillon Cullinan`: https://github.com/dillon-cullinan
.. _`Matthieu Bulte`: https://github.com/matthieubulte
.. _`Olli Koskinen`: https://github.com/okoskinen
.. _`Markku Luukkainen`: https://github.com/mluukkainen
.. _`Sangeeth Keeriyadath`: https://github.com/ksangeek
