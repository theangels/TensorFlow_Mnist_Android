package org.ubilabs.angel.androiddemo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    private TextView displayNumber;

    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private static final String MODEL_FILE = "file:///android_asset/mnist_model_graph.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/graph_label_strings.txt";

    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();


    @SuppressWarnings("SuspiciousNameCombination")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        displayNumber = (TextView) findViewById(R.id.displayNumber);

        initTensorFlowAndLoadModel();
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    Log.d(TAG, "Load Success");
                    doReconize();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void doReconize() {
        try {
            InputStream is = getResources().getAssets().open("test.png");
            Bitmap bitmap  = BitmapFactory.decodeStream(is);

            int[] pixelsInt = new int[INPUT_SIZE * INPUT_SIZE];
            float[] pixels = new float[INPUT_SIZE * INPUT_SIZE];

            bitmap.getPixels(pixelsInt, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            for (int i = 0; i < pixelsInt.length; i++) {
                pixels[i] = (255-pixelsInt[i]) * 1f/255;
            }

            final List<Classifier.Recognition> results = classifier.recognizeImage(pixels);

            if (results.size() > 0) {
                String value = " Number is : " + results.get(0).getTitle();
                displayNumber.setText(value);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
