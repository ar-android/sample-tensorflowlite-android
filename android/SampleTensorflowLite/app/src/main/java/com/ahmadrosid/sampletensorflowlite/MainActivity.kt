package com.ahmadrosid.sampletensorflowlite

import android.content.Intent
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import com.ahmadrosid.sampletensorflowlite.camera.CameraActivity
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnOpenCamera.setOnClickListener {
            val intent = Intent(this, CameraActivity::class.java)
            startActivity(intent)
        }
    }
}
