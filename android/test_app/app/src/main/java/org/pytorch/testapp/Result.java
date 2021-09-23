package org.pytorch.testapp;

import java.util.List;

class Result {
  public final int tensorSize;
  public final List<BBox> bboxes;
  public final long totalDuration;
  public final long moduleForwardDuration;

  public Result(int tensorSize, List<BBox> bboxes, long moduleForwardDuration, long totalDuration) {
    this.tensorSize = tensorSize;
    this.bboxes = bboxes;
    this.moduleForwardDuration = moduleForwardDuration;
    this.totalDuration = totalDuration;
  }
}
