Needs to be done in order to migrate to Beta
--------------------------------------------

(Some of the items really just mean "Nicolas needs to understand this better")

* Figure out logistics of migration (extra .v2 namespace, programmatic "opt-in",
  stuff like that): tracked in https://github.com/pytorch/vision/issues/7097
* Figure out dataset <-> transformsV2 layer (including HF or other external
  datasets): tracked in https://github.com/pytorch/vision/pull/6663
* Figure out internal video partners and what they actually need. Some of the
  Video transforms like `uniform_temporal_subsample()` are outliers (awkward
  support, doesn't fit well into the current API). Same for `PermuteDimensions`
  and `TransposeDimension` which break underlying assumptions about dimension
  order.
* Address critical TODOs below and in code, code review etc.
* Write Docs

Needs to be done before migrating to stable
-------------------------------------------

* Polish tests - make sure they are at least functionally equivalent to the v1
  tests. This requires individually checking them.
* Address rest of TODOs below and in code, code review etc.
* Look into pytorch 2.0 compat? (**Should this be bumped up??**)
* Figure out path to user-defined transforms and sub-classes 
* Add support for Optical Flow tranforms (e.g. vlip needs special handling for
  flow masks)
* Handle strides, e.g. https://github.com/pytorch/vision/issues/7090 ? Looks like it's a non-issue?
* Figure out what transformsV2 mean for inference presets


TODOs
-----

- Those in https://github.com/pytorch/vision/pull/7092 and
  https://github.com/pytorch/vision/pull/7082 (There is overlap!)
  They're not all critical.
- Document (internally, not as user-facing docs) the `self.as_subclass(torch.Tensor)` perf hack 

Done
----

* Figure out what to do about get_params() static methods (See https://github.com/pytorch/vision/pull/7092).
  A: we want them back - tracked in https://github.com/pytorch/vision/pull/7153
* Avoid inconsistent output type: Let Normalize() and RandomPhotometricDistort
  return datapoints instead of tensors
  (https://github.com/pytorch/vision/pull/7113)
* Figure out criticality of JIT compat for classes. Is this absolutely needed,
  by whom, potential workarounds, etc.
  * Done: Philip found elegant way to support JIT as long as the v1 transforms
    are still around: https://github.com/pytorch/vision/pull/7135
* Figure out whether `__torch_dispatch__` is preferable to `__torch_function__`.
  * After chat with Alban, there's no reason to use `__torch_dispatch__`.
    Everything should work as expected with `__torch_function__`, including
    AutoGrad.
* Simplify interface and Image meta-data: Remove color_space metadata and
  ConvertColorSpace() transform (https://github.com/pytorch/vision/pull/7120)