package com.example.airqualitychecker;
import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.content.Intent;
import android.widget.TextView;

import java.text.DecimalFormat;

public class ResultActivity extends  AppCompatActivity{
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.result_page);

        Button btnBackResult = findViewById(R.id.idBtnBackResult);
        TextView result = findViewById(R.id.tvResult);
        TextView tvShow = findViewById(R.id.tvShow);

        String aqiCategory;
        Intent intent = getIntent();
        float predictedAqi = getIntent().getFloatExtra("predicted_aqi", 0.0f);
        DecimalFormat decimalFormat = new DecimalFormat("#.##");
        String formattedPredictedAqi = decimalFormat.format(predictedAqi);

        result.setText("Predicted AQI: " + formattedPredictedAqi);

        // Set the background color based on the predicted AQI value
        if (predictedAqi <= 50) {
            result.setBackgroundColor(getResources().getColor(R.color.green)); // Set background color to green
        } else if (predictedAqi < 100) {
            result.setBackgroundColor(getResources().getColor(R.color.yellow)); // Set background color to yellow
        } else {
            result.setBackgroundColor(getResources().getColor(R.color.red)); // Set background color to red
        }


        if (predictedAqi <= 50) {
            // Good AQI
            aqiCategory = "ভালো";
            tvShow.setTextColor(getResources().getColor(R.color.green));
        } else if (predictedAqi <= 100) {
            // Moderate AQI
            aqiCategory = "সতর্কতামূলক";
            tvShow.setTextColor(getResources().getColor(R.color.yellow));
        } else {
            // Unhealthy AQI
            aqiCategory = "অস্বাস্থ্যকর";
            tvShow.setTextColor(getResources().getColor(R.color.red));
        }

// Set the text of the TextView
        tvShow.setText(aqiCategory);

        btnBackResult.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });

    }

}
