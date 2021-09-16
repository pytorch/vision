---
title: Scheduled workflow failed
labels:
  - bug
  - "module: datasets"
---

Oh no, something went wrong in the scheduled workflow {{ env.WORKFLOW }}/{{ env.JOB }}. 
Please look into it:

https://github.com/{{ env.REPO }}/actions/runs/{{ env.ID }}

Feel free to close this if this was just a one-off error.
