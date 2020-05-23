package com.example.cifar10camera;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private TextView resultsTextView;
    private Bitmap bitmap;
    static final int REQUEST_IMAGE_CAPTURE = 1;
    static final int REQUEST_CHOOSE_IMAGE_GALLERY = 2;

    static {
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_FILE = "file:///android_asset/cifar10.pb";
    private static final String INPUT_NODE = "reshape_1_input";
    private static final long[] INPUT_SHAPE = {1, 3072};
    private static final String OUTPUT_NODE = "dense_2/Softmax";
    TensorFlowInferenceInterface inferenceInterface;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

        imageView = findViewById(R.id.image);
        resultsTextView = findViewById(R.id.results);
    }

    public void getImageFromCamera(View view) {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    public void getImageFromGallery(View view) {
        if(ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_CHOOSE_IMAGE_GALLERY);
        }
        else {
            getImageFromGallery();
        }
    }

    private void getImageFromGallery() {
        Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(pickPhoto, REQUEST_CHOOSE_IMAGE_GALLERY);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == REQUEST_CHOOSE_IMAGE_GALLERY) {
            if(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                getImageFromGallery();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            bitmap = (Bitmap) extras.get("data");
            imageView.setImageBitmap(bitmap);
        }
        else if(requestCode == REQUEST_CHOOSE_IMAGE_GALLERY && resultCode == RESULT_OK) {
            try {
                Uri selectedImage = data.getData();
                InputStream imageStream = getContentResolver().openInputStream(selectedImage);
                bitmap = BitmapFactory.decodeStream(imageStream);
                imageView.setImageBitmap(bitmap);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public void guessImageAction(View view) {
        if(bitmap != null) {
            float[] pixelBuffer = formatImageData();
            float[] results = makePredictions(pixelBuffer);
            displayResults(results);
        }
        else {
            Toast.makeText(this, "First Choose Image", Toast.LENGTH_LONG).show();
        }
    }

    private float[] formatImageData() {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 32, 32, true);
        int[] intArray = new int[1024];
        scaledBitmap.getPixels(intArray, 0, 32, 0,0,32, 32);
        float[] floatArray = new float[3072];

        for(int i = 0; i < 1024; ++i) {
            floatArray[i] = ((intArray[i] >> 16) & 0xff) / 255.0f;
            floatArray[i + 1] = ((intArray[i] >> 8) & 0xff) / 255.0f;
            floatArray[i + 2] = (intArray[i] & 0xff) / 255.0f;
        }
        return floatArray;
    }

    private float[] makePredictions(float[] pixelBuffer) {
        float[] results = new float[10];
        inferenceInterface.feed(INPUT_NODE, pixelBuffer, INPUT_SHAPE);
        inferenceInterface.run(new String[] {OUTPUT_NODE});
        inferenceInterface.fetch(OUTPUT_NODE, results);
        return results;
    }

    private void displayResults(float[] resultsArray) {
        String[] answers =  {
                "Airplane",
                "Automobile",
                "Bird",
                "Cat",
                "Deer",
                "Dog",
                "Frog",
                "Horse",
                "Ship",
                "Truck",
        };

        int maxIndex = 0, secondMaxIndex = 0;
        float max = 0, secondMax = 0;
        for (int i = 0; i < 10; ++i) {
            if(resultsArray[i] > max) {
                secondMax = max;
                secondMaxIndex = maxIndex;
                maxIndex = i;
                max = resultsArray[i];
            }
            else if(resultsArray[i] > secondMax) {
                secondMaxIndex = i;
                secondMax = resultsArray[i];
            }
        }

        resultsTextView.setText("Prediction:\t" + answers[maxIndex] + "\nSecond Prediction:\t" + answers[secondMaxIndex]);
    }
}
