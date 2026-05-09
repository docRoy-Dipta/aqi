package com.example.airqualitychecker;

import android.app.DatePickerDialog;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.util.Calendar;

import okhttp3.FormBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class InputActivity extends AppCompatActivity{
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.input_page);

        Button pickDateBtn = findViewById(R.id.idBtnPickDate);
        TextView selectedDateTV = findViewById(R.id.idTVSelectedDate);
        Button predictBtn= findViewById(R.id.idBtnPredict);
        EditText etNow = findViewById(R.id.etNow);
        EditText etRaw = findViewById(R.id.etRaw);

        etNow.setText("");
        etRaw.setText("");
        selectedDateTV.setText("");


        pickDateBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // on below line we are getting
                // the instance of our calendar.
                final Calendar c = Calendar.getInstance();

                // on below line we are getting
                // our day, month and year.
                int year = c.get(Calendar.YEAR);
                int month = c.get(Calendar.MONTH);
                int day = c.get(Calendar.DAY_OF_MONTH);

                // on below line we are creating a variable for date picker dialog.
                DatePickerDialog datePickerDialog = new DatePickerDialog(
                        // on below line we are passing context.
                        InputActivity.this,
                        new DatePickerDialog.OnDateSetListener() {
                            @Override
                            public void onDateSet(DatePicker view, int year,
                                                  int monthOfYear, int dayOfMonth) {
                                // on below line we are setting date to our text view.
                                selectedDateTV.setText(dayOfMonth + "-" + (monthOfYear + 1) + "-" + year);

                            }
                        },
                        // on below line we are passing year,
                        // month and day for selected date in our date picker.
                        year, month, day);
                // at last we are calling show to
                // display our date picker dialog.
                datePickerDialog.show();
            }
        });

        predictBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                // Get input values
                String nowcast = etNow.getText().toString();
                String raw = etRaw.getText().toString();
                String date = selectedDateTV.getText().toString();

                // Validate input fields
                if (!nowcast.isEmpty() && !raw.isEmpty() && !date.isEmpty()) {
                    // Execute the AsyncTask
                    new PredictionTask().execute(nowcast, raw, date);
                } else {
                    // Display a Toast message indicating all fields must be filled
                    Toast.makeText(InputActivity.this, "All fields must be filled", Toast.LENGTH_SHORT).show();
                }

            }
        });


        Button btnBackInput = findViewById(R.id.idBtnBackInput);

        btnBackInput.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(InputActivity.this, MainActivity.class);
                startActivity(i);
                finish();
            }
        });
    }
    private class PredictionTask extends AsyncTask<String, Void, Float> {

        @Override
        protected Float doInBackground(String... params) {
            String nowcast = params[0];
            String raw = params[1];
            String date = params[2];

            OkHttpClient client = new OkHttpClient();

            // Define your server URL with the correct IP address
            String serverUrl = "http://13.48.26.38/predict";

            // Create a form object to send as parameters
            FormBody formBody = new FormBody.Builder()
                    .add("nowcast", nowcast)
                    .add("raw", raw)
                    .add("date", date)
                    .build();

            RequestBody requestBody = formBody;

            Request request = new Request.Builder()
                    .url(serverUrl)
                    .post(requestBody)
                    .build();

            try {
                Response response = client.newCall(request).execute();
                if (response.isSuccessful()) {
                    String responseBody = response.body().string();
                    response.close();
                    try {
                        // Parse the response value as a float
                        return Float.parseFloat(responseBody);
                    } catch (NumberFormatException e) {
                        Log.e("PARSE_ERROR", "Error parsing predicted AQI value: " + e.getMessage());
                    }
                } else {
                    // Handle unsuccessful response
                    Log.e("HTTP_REQUEST_ERROR", "Error: " + response.message());
                }
            } catch (IOException e) {
                // Handle errors here
                Log.e("HTTP_REQUEST_ERROR", "Error: " + e.getMessage());
            }
            client.connectionPool().evictAll();
            return null;
        }

        @Override
        protected void onPostExecute(Float predictedAqi) {
            super.onPostExecute(predictedAqi);
            if (predictedAqi != null) {
                // Start ResultActivity and pass the prediction value
                Intent intent = new Intent(InputActivity.this, ResultActivity.class);
                intent.putExtra("predicted_aqi", predictedAqi);
                startActivity(intent);
            } else {
                // Handle unsuccessful response
                Log.e("HTTP_REQUEST_ERROR", "Error: ");
            }
        }
    }
}
