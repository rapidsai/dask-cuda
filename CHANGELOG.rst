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
