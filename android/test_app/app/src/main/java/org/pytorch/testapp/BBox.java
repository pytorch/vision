package org.pytorch.testapp;

class BBox {
  public final float score;
  public final float x0;
  public final float y0;
  public final float x1;
  public final float y1;

  public BBox(float score, float x0, float y0, float x1, float y1) {
    this.score = score;
    this.x0 = x0;
    this.y0 = y0;
    this.x1 = x1;
    this.y1 = y1;
  }

  @Override
  public String toString() {
    return String.format("Box{score=%f x0=%f y0=%f x1=%f y1=%f", score, x0, y0, x1, y1);
  }
}
