package org.pytorch.testapp;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.view.ViewStub;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

public class CameraActivity extends AppCompatActivity {

  private static final float BBOX_SCORE_DRAW_THRESHOLD = 0.5f;
  private static final String TAG = BuildConfig.LOGCAT_TAG;
  private static final int TEXT_TRIM_SIZE = 4096;
  private static final int RGB_MAX_CHANNEL_VALUE = 262143;

  private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
  private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni");
    NativeLoader.loadLibrary("torchvision_ops");
  }

  private Bitmap mInputTensorBitmap;
  private Bitmap mBitmap;
  private Canvas mCanvas;

  private long mLastAnalysisResultTime;

  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;
  protected Handler mUIHandler;

  private TextView mTextView;
  private ImageView mCameraOverlay;
  private StringBuilder mTextViewStringBuilder = new StringBuilder();

  private Paint mBboxPaint;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_camera);
    mTextView = findViewById(R.id.text);
    mCameraOverlay = findViewById(R.id.camera_overlay);
    mUIHandler = new Handler(getMainLooper());
    startBackgroundThread();

    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_CODE_CAMERA_PERMISSION);
    } else {
      setupCameraX();
    }
    mBboxPaint = new Paint();
    mBboxPaint.setAntiAlias(true);
    mBboxPaint.setDither(true);
    mBboxPaint.setColor(Color.GREEN);
  }

  @Override
  protected void onPostCreate(@Nullable Bundle savedInstanceState) {
    super.onPostCreate(savedInstanceState);
    startBackgroundThread();
  }

  protected void startBackgroundThread() {
    mBackgroundThread = new HandlerThread("ModuleActivity");
    mBackgroundThread.start();
    mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
  }

  @Override
  protected void onDestroy() {
    stopBackgroundThread();
    super.onDestroy();
  }

  protected void stopBackgroundThread() {
    mBackgroundThread.quitSafely();
    try {
      mBackgroundThread.join();
      mBackgroundThread = null;
      mBackgroundHandler = null;
    } catch (InterruptedException e) {
      Log.e(TAG, "Error on stopping background thread", e);
    }
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
      if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
        Toast.makeText(
                this,
                "You can't use image classification example without granting CAMERA permission",
                Toast.LENGTH_LONG)
            .show();
        finish();
      } else {
        setupCameraX();
      }
    }
  }

  private void setupCameraX() {
    final TextureView textureView =
        ((ViewStub) findViewById(R.id.camera_texture_view_stub))
            .inflate()
            .findViewById(R.id.texture_view);
    final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
    final Preview preview = new Preview(previewConfig);
    preview.setOnPreviewOutputUpdateListener(
        new Preview.OnPreviewOutputUpdateListener() {
          @Override
          public void onUpdated(Preview.PreviewOutput output) {
            textureView.setSurfaceTexture(output.getSurfaceTexture());
          }
        });

    final DisplayMetrics displayMetrics = new DisplayMetrics();
    getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);

    final ImageAnalysisConfig imageAnalysisConfig =
        new ImageAnalysisConfig.Builder()
            .setTargetResolution(new Size(displayMetrics.widthPixels, displayMetrics.heightPixels))
            .setCallbackHandler(mBackgroundHandler)
            .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            .build();
    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    imageAnalysis.setAnalyzer(
        new ImageAnalysis.Analyzer() {
          @Override
          public void analyze(ImageProxy image, int rotationDegrees) {
            if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500) {
              return;
            }

            final Result result = CameraActivity.this.analyzeImage(image, rotationDegrees);

            if (result != null) {
              mLastAnalysisResultTime = SystemClock.elapsedRealtime();
              CameraActivity.this.runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      CameraActivity.this.handleResult(result);
                    }
                  });
            }
          }
        });

    CameraX.bindToLifecycle(this, preview, imageAnalysis);
  }

  private Module mModule;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;

  private static int clamp0255(int x) {
    if (x > 255) {
      return 255;
    }
    return x < 0 ? 0 : x;
  }

  protected void fillInputTensorBuffer(
      ImageProxy image, int rotationDegrees, FloatBuffer inputTensorBuffer) {

    if (mInputTensorBitmap == null) {
      final int tensorSize = Math.min(image.getWidth(), image.getHeight());
      mInputTensorBitmap = Bitmap.createBitmap(tensorSize, tensorSize, Bitmap.Config.ARGB_8888);
    }

    ImageProxy.PlaneProxy[] planes = image.getPlanes();
    ImageProxy.PlaneProxy Y = planes[0];
    ImageProxy.PlaneProxy U = planes[1];
    ImageProxy.PlaneProxy V = planes[2];
    ByteBuffer yBuffer = Y.getBuffer();
    ByteBuffer uBuffer = U.getBuffer();
    ByteBuffer vBuffer = V.getBuffer();
    final int imageWidth = image.getWidth();
    final int imageHeight = image.getHeight();
    final int tensorSize = Math.min(imageWidth, imageHeight);

    int widthAfterRtn = imageWidth;
    int heightAfterRtn = imageHeight;
    boolean oddRotation = rotationDegrees == 90 || rotationDegrees == 270;
    if (oddRotation) {
      widthAfterRtn = imageHeight;
      heightAfterRtn = imageWidth;
    }

    int minSizeAfterRtn = Math.min(heightAfterRtn, widthAfterRtn);
    int cropWidthAfterRtn = minSizeAfterRtn;
    int cropHeightAfterRtn = minSizeAfterRtn;

    int cropWidthBeforeRtn = cropWidthAfterRtn;
    int cropHeightBeforeRtn = cropHeightAfterRtn;
    if (oddRotation) {
      cropWidthBeforeRtn = cropHeightAfterRtn;
      cropHeightBeforeRtn = cropWidthAfterRtn;
    }

    int offsetX = (int) ((imageWidth - cropWidthBeforeRtn) / 2.f);
    int offsetY = (int) ((imageHeight - cropHeightBeforeRtn) / 2.f);

    int yRowStride = Y.getRowStride();
    int yPixelStride = Y.getPixelStride();
    int uvRowStride = U.getRowStride();
    int uvPixelStride = U.getPixelStride();

    float scale = cropWidthAfterRtn / tensorSize;
    int yIdx, uvIdx, yi, ui, vi;
    final int channelSize = tensorSize * tensorSize;
    for (int y = 0; y < tensorSize; y++) {
      for (int x = 0; x < tensorSize; x++) {
        final int centerCropX = (int) Math.floor(x * scale);
        final int centerCropY = (int) Math.floor(y * scale);
        int srcX = centerCropX + offsetX;
        int srcY = centerCropY + offsetY;

        if (rotationDegrees == 90) {
          srcX = offsetX + centerCropY;
          srcY = offsetY + (minSizeAfterRtn - 1) - centerCropX;
        } else if (rotationDegrees == 180) {
          srcX = offsetX + (minSizeAfterRtn - 1) - centerCropX;
          srcY = offsetY + (minSizeAfterRtn - 1) - centerCropY;
        } else if (rotationDegrees == 270) {
          srcX = offsetX + (minSizeAfterRtn - 1) - centerCropY;
          srcY = offsetY + centerCropX;
        }

        yIdx = srcY * yRowStride + srcX * yPixelStride;
        uvIdx = (srcY >> 1) * uvRowStride + (srcX >> 1) * uvPixelStride;

        yi = yBuffer.get(yIdx) & 0xff;
        ui = uBuffer.get(uvIdx) & 0xff;
        vi = vBuffer.get(uvIdx) & 0xff;

        yi = (yi - 16) < 0 ? 0 : (yi - 16);
        ui -= 128;
        vi -= 128;

        int a0 = 1192 * yi;
        int ri = (a0 + 1634 * vi);
        int gi = (a0 - 833 * vi - 400 * ui);
        int bi = (a0 + 2066 * ui);

        ri = ri > RGB_MAX_CHANNEL_VALUE ? RGB_MAX_CHANNEL_VALUE : (ri < 0 ? 0 : ri);
        gi = gi > RGB_MAX_CHANNEL_VALUE ? RGB_MAX_CHANNEL_VALUE : (gi < 0 ? 0 : gi);
        bi = bi > RGB_MAX_CHANNEL_VALUE ? RGB_MAX_CHANNEL_VALUE : (bi < 0 ? 0 : bi);

        final int color =
            0xff000000 | ((ri << 6) & 0xff0000) | ((gi >> 2) & 0xff00) | ((bi >> 10) & 0xff);
        mInputTensorBitmap.setPixel(x, y, color);
        inputTensorBuffer.put(0 * channelSize + y * tensorSize + x, clamp0255(ri >> 10) / 255.f);
        inputTensorBuffer.put(1 * channelSize + y * tensorSize + x, clamp0255(gi >> 10) / 255.f);
        inputTensorBuffer.put(2 * channelSize + y * tensorSize + x, clamp0255(bi >> 10) / 255.f);
      }
    }
  }

  public static String assetFilePath(Context context, String assetName) {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    } catch (IOException e) {
      Log.e(TAG, "Error process asset " + assetName + " to file path");
    }
    return null;
  }

  @WorkerThread
  @Nullable
  protected Result analyzeImage(ImageProxy image, int rotationDegrees) {
    Log.i(TAG, String.format("analyzeImage(%s, %d)", image, rotationDegrees));
    final int tensorSize = Math.min(image.getWidth(), image.getHeight());
    if (mModule == null) {
      Log.i(TAG, "Loading module from asset '" + BuildConfig.MODULE_ASSET_NAME + "'");
      mInputTensorBuffer = Tensor.allocateFloatBuffer(3 * tensorSize * tensorSize);
      mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[] {3, tensorSize, tensorSize});
      final String modelFileAbsoluteFilePath =
          new File(assetFilePath(this, BuildConfig.MODULE_ASSET_NAME)).getAbsolutePath();
      mModule = Module.load(modelFileAbsoluteFilePath);
    }

    final long startTime = SystemClock.elapsedRealtime();
    fillInputTensorBuffer(image, rotationDegrees, mInputTensorBuffer);

    final long moduleForwardStartTime = SystemClock.elapsedRealtime();
    final IValue outputTuple = mModule.forward(IValue.listFrom(mInputTensor));
    final IValue out1 = outputTuple.toTuple()[1];
    final Map<String, IValue> map = out1.toList()[0].toDictStringKey();

    float[] boxesData = new float[] {};
    float[] scoresData = new float[] {};
    final List<BBox> bboxes = new ArrayList<>();
    if (map.containsKey("boxes")) {
      final Tensor boxesTensor = map.get("boxes").toTensor();
      final Tensor scoresTensor = map.get("scores").toTensor();
      boxesData = boxesTensor.getDataAsFloatArray();
      scoresData = scoresTensor.getDataAsFloatArray();
      final int n = scoresData.length;
      for (int i = 0; i < n; i++) {
        final BBox bbox =
            new BBox(
                scoresData[i],
                boxesData[4 * i + 0],
                boxesData[4 * i + 1],
                boxesData[4 * i + 2],
                boxesData[4 * i + 3]);
        android.util.Log.i(TAG, String.format("Forward result %d: %s", i, bbox));
        bboxes.add(bbox);
      }
    } else {
      android.util.Log.i(TAG, "Forward result empty");
    }

    final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;
    final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
    return new Result(tensorSize, bboxes, moduleForwardDuration, analysisDuration);
  }

  @UiThread
  protected void handleResult(Result result) {
    final int W = mCameraOverlay.getMeasuredWidth();
    final int H = mCameraOverlay.getMeasuredHeight();

    final int size = Math.min(W, H);
    final int offsetX = (W - size) / 2;
    final int offsetY = (H - size) / 2;

    float scaleX = (float) size / result.tensorSize;
    float scaleY = (float) size / result.tensorSize;
    if (mBitmap == null) {
      mBitmap = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888);
      mCanvas = new Canvas(mBitmap);
    }

    mCanvas.drawBitmap(
        mInputTensorBitmap,
        new Rect(0, 0, result.tensorSize, result.tensorSize),
        new Rect(offsetX, offsetY, offsetX + size, offsetY + size),
        null);

    for (final BBox bbox : result.bboxes) {
      if (bbox.score < BBOX_SCORE_DRAW_THRESHOLD) {
        continue;
      }

      float c_x0 = offsetX + scaleX * bbox.x0;
      float c_y0 = offsetY + scaleY * bbox.y0;

      float c_x1 = offsetX + scaleX * bbox.x1;
      float c_y1 = offsetY + scaleY * bbox.y1;

      mCanvas.drawLine(c_x0, c_y0, c_x1, c_y0, mBboxPaint);
      mCanvas.drawLine(c_x1, c_y0, c_x1, c_y1, mBboxPaint);
      mCanvas.drawLine(c_x1, c_y1, c_x0, c_y1, mBboxPaint);
      mCanvas.drawLine(c_x0, c_y1, c_x0, c_y0, mBboxPaint);
      mCanvas.drawText(String.format("%.2f", bbox.score), c_x0, c_y0, mBboxPaint);
    }
    mCameraOverlay.setImageBitmap(mBitmap);

    String message = String.format("forwardDuration:%d", result.moduleForwardDuration);
    Log.i(TAG, message);
    mTextViewStringBuilder.insert(0, '\n').insert(0, message);
    if (mTextViewStringBuilder.length() > TEXT_TRIM_SIZE) {
      mTextViewStringBuilder.delete(TEXT_TRIM_SIZE, mTextViewStringBuilder.length());
    }
    mTextView.setText(mTextViewStringBuilder.toString());
  }
}
