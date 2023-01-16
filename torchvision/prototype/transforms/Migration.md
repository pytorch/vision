Needs to be done in order to migrate to stable
----------------------------------------------

(Some of the items really just mean "Nicolas needs to understand this better")

* Figure out criticality of JIT compat for classes. Is this absolutely needed, by whom, potential workarounds, etc.
* Write Docs
* Figure out dataset <-> transformsV2 layer (including HF or other external datasets)
* address TODOs below and in code, code review etc.
* pytorch 2.0 compat?
* Video transforms
* Figure out path to user-defined transforms and sub-classes 
* Does Mask support Optical Flow masks?
* Handle strides, e.g. https://github.com/pytorch/vision/issues/7090 ? Looks like it's a non-issue?
* Figure out what transformsV2 mean for inference presets
* Figure out logistics of migration into stable area (extra .v2 namespace, stuff like that)



TODOs
-----

- Those in https://github.com/pytorch/vision/pull/7092 and
  https://github.com/pytorch/vision/pull/7082 (There is overlap!)
  They're not all critical.
- Document (internally, not as user-facing docs) the `self.as_subclass(torch.Tensor)` perf hack 

Done
----

* Figure out what to do about get_params() static methods (See https://github.com/pytorch/vision/pull/7092)