package org.pytorch.testapp;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.nio.FloatBuffer;
import java.util.Map;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;

public class MainActivity extends AppCompatActivity {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni");
    NativeLoader.loadLibrary("torchvision_ops");
  }

  private static final String TAG = BuildConfig.LOGCAT_TAG;
  private static final int TEXT_TRIM_SIZE = 4096;

  private TextView mTextView;

  protected HandlerThread mBackgroundThread;
  protected Handler mBackgroundHandler;
  private Module mModule;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private StringBuilder mTextViewStringBuilder = new StringBuilder();

  private final Runnable mModuleForwardRunnable =
      new Runnable() {
        @Override
        public void run() {
          final Result result = doModuleForward();
          runOnUiThread(
              () -> {
                handleResult(result);
                if (mBackgroundHandler != null) {
                  mBackgroundHandler.post(mModuleForwardRunnable);
                }
              });
        }
      };

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    mTextView = findViewById(R.id.text);
    startBackgroundThread();
    mBackgroundHandler.post(mModuleForwardRunnable);
  }

  protected void startBackgroundThread() {
    mBackgroundThread = new HandlerThread(TAG + "_bg");
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
      Log.e(TAG, "Error stopping background thread", e);
    }
  }

  @WorkerThread
  @Nullable
  protected Result doModuleForward() {
    if (mModule == null) {
      final long[] shape = BuildConfig.INPUT_TENSOR_SHAPE;
      long numElements = 1;
      for (int i = 0; i < shape.length; i++) {
        numElements *= shape[i];
      }
      mInputTensorBuffer = Tensor.allocateFloatBuffer((int) numElements);
      mInputTensor = Tensor.fromBlob(mInputTensorBuffer, BuildConfig.INPUT_TENSOR_SHAPE);
      PyTorchAndroid.setNumThreads(1);
      mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), BuildConfig.MODULE_ASSET_NAME);
    }

    final long startTime = SystemClock.elapsedRealtime();
    final long moduleForwardStartTime = SystemClock.elapsedRealtime();
    final IValue outputTuple = mModule.forward(IValue.listFrom(mInputTensor));
    final IValue[] outputArray = outputTuple.toTuple();
    final IValue out0 = outputArray[0];
    final Map<String, IValue> map = out0.toDictStringKey();
    if (map.containsKey("boxes")) {
      final Tensor boxes = map.get("boxes").toTensor();
      final Tensor scores = map.get("scores").toTensor();
      final float[] boxesData = boxes.getDataAsFloatArray();
      final float[] scoresData = scores.getDataAsFloatArray();
      final int n = scoresData.length;
      for (int i = 0; i < n; i++) {
        android.util.Log.i(
            TAG,
            String.format(
                "Forward result %d: score %f box:(%f, %f, %f, %f)",
                scoresData[i],
                boxesData[4 * i + 0],
                boxesData[4 * i + 1],
                boxesData[4 * i + 2],
                boxesData[4 * i + 3]));
      }
    } else {
      android.util.Log.i(TAG, "Forward result empty");
    }

    final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;
    final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
    return new Result(new float[] {}, moduleForwardDuration, analysisDuration);
  }

  static class Result {

    private final float[] scores;
    private final long totalDuration;
    private final long moduleForwardDuration;

    public Result(float[] scores, long moduleForwardDuration, long totalDuration) {
      this.scores = scores;
      this.moduleForwardDuration = moduleForwardDuration;
      this.totalDuration = totalDuration;
    }
  }

  @UiThread
  protected void handleResult(Result result) {
    String message = String.format("forwardDuration:%d", result.moduleForwardDuration);
    mTextViewStringBuilder.insert(0, '\n').insert(0, message);
    if (mTextViewStringBuilder.length() > TEXT_TRIM_SIZE) {
      mTextViewStringBuilder.delete(TEXT_TRIM_SIZE, mTextViewStringBuilder.length());
    }
    mTextView.setText(mTextViewStringBuilder.toString());
  }
}
