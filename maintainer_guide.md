## Torchvision maintainers guide

This document aims at documenting common questions and workflows that arise when
developing and maintaining torchvision. It does not serve as an absolute ground
truth, but rather as a handy reference guide.

### What is public?

* In the torchvision folder:
    * For Python, the usual convention applies: anything that has a leading
      underscore in its name or in its path is private. The rest is implicitly
      public, even if it’s not properly documented, and even if it’s not exposed
      in an `__init__` file.
    * For C++, code is private. If a change breaks fbcode, fix fbcode or revert
      the change. We should be careful about models running in prod and relying
      on torchvision ops. FYI PyTorch as BC API checks for its ops.
* The test folder is not importable and is **private.** Even fbcode projects
  should not rely on it. 
* The references folder serves as documentation and is treated as such: it’s
  “private” in the sense that there are no BC guarantees here. Breaking changes
  are possible, but we should make sure that the tutorials are still running
  properly, and that their intended narrative is preserved (by e.g. checking
  outputs, etc.).
* The rest of the folders (build, android, ios) is private and has no BC
  breaking guarantees.

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
  There might be a ~1% left that’s not addressed: that’s OK. Also, make what’s
  wrong very hard, if not impossible.
* For models, things are more subtle as we still need a way to properly enable
  model surgery. TODO: expand on this

As a good practice, always create new files and even classes with a leading
underscore in their name. This way, everything is private by default and the
only public surface is explicitly present in an `__init__` file.

### Should we deprecate X, or can we make a BC-breaking change?

When a public-facing feature needs to change, we try to deprecate, but sometimes
a BC-breaking change is possible or even warranted. Typically in the case of
bugs, a failure is better than incorrect results. To decide whether to deprecate
or to introduce a breaking change:

1. Check whether X does something purely internal VS something that a reasonable
   user might be interested in.
2. Look for usages on Github for usages of X. This is often difficult due to the
   forks, copy-pastes, and name conflicts.
3. Look on FBcode where we can get a cleaner (but also biased) signal of the
   usage.
4. Weigh the pros and cons of breaking. When there is a bug involved, it’s often
   OK to break, as we want to prevent users from doing wrong things or having
   wrong results. See this example:
   [_https://github.com/pytorch/vision/pull/2954_](https://github.com/pytorch/vision/pull/2954)
5. Debate this openly among people of the team and community, and adapt the
   decision based on the feedback.

Either way, make sure to let users know how to work around and adapt to the new
change. For deprecations, this can often be done as part of the deprecation
warning message, **and** as part of the docstring with the[_..
deprecated::_](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-deprecated)
directive. For BC breaking changes, it can be directly embedded in the release
notes for simple cases, or the notes can link to a comment in the PR with more
detailed instructions.
