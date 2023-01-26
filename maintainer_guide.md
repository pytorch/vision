## Torchvision maintainers guide

This document aims at documenting user-facing policies / principles used when
developing and maintaining torchvision. Other maintainer info (e.g. release
process) can be found in the meta-internal wiki.

### What is public and what is private?

For the Python API, torchvision largely follows the [PyTorch
policy](https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation)
which is consistent with other major packages
([numpy](https://numpy.org/neps/nep-0023-backwards-compatibility.html),
[scikit-learn](https://scikit-learn.org/dev/glossary.html#term-API) etc.).
We recognize that his policy is somewhat imperfect for some edge cases, and that
it's difficult to come up with an accurate technical definition. In broad terms,
which are usually well understood by users, the policy is that:

- modules that can be accessed without leading underscore are public
- objects in a public file that don't have a leading underscore are public
- class attributes are public iff they have no leading underscore
- the rest of the modules / objects / class attributes are considered private

The public API has backward-compatible (BC) guarantees defined in our
deprecation policy (see below). The private API has not BC guarantees.

For C++, code is private. For Meta employees: if a C++ change breaks fbcode, fix
fbcode or revert the change. We should be careful about models running in
production and relying on torchvision ops.

The `test` folder is not importable and is **private.** Even meta-internal
projects should *not* rely on it (it has happened in the past and is now
programmatically impossible).

The training references do not have BC guarantees. Breaking changes are
possible, but we should make sure that the tutorials are still running properly,
and that their intended narrative is preserved (by e.g. checking outputs,
etc.).

The rest of the folders (build, android, ios, etc.) are private and have no BC
guarantees.

### Deprecation policy.

Because they're disruptive, **deprecations should only be used sparingly**.

We largely follow the [PyTorch
policy](https://github.com/pytorch/pytorch/wiki/PyTorch's-Python-Frontend-Backward-and-Forward-Compatibility-Policy):
breaking changes require a deprecation period of at least 2 versions.

Deprecations should clearly indicate their deadline in the docs and warning
messages. Avoid not committing to a deadline, or keeping deprecated APIs for too
long: it gives no incentive for users to update their code, sends conflicting
messages ("why was this API removed while this other one is still around?"), and
accumulates debt in the project.

### Should this attribute be public? Should this function be private?

When designing an API it’s not always obvious what should be exposed as public,
and what should be kept as a private implementation detail. The following
guidelines can be useful:

* Functional consistency throughout the library is a top priority, for users and
  developers’ sake. In doubt and unless it’s clearly wrong, expose what other
  similar classes expose.
* Think really hard about the users and their use-cases, and try to expose what
  they would need to address those use-cases. Aggressively keep everything else
  private. Remember that the “private -> public” direction is way smoother than
  the “public -> private” one: in doubt, keep it private.
* When thinking about use-cases, the general API motto applies: make what’s
  simple and common easy, and make what’s complex possible (80% / 20% rule).
  There might be a ~1% left that’s not addressed: that’s OK. Also, **make what’s
  wrong very hard**, if not impossible.

As a good practice, always create new files and even classes with a leading
underscore in their name. This way, everything is private by default and the
only public surface is explicitly present in an `__init__.py` file.
