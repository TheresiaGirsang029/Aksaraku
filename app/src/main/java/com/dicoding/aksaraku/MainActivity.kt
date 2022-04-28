package com.dicoding.aksaraku

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.dicoding.aksaraku.databinding.ActivityMainBinding
import com.dicoding.aksaraku.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var imageView: ImageView
    private lateinit var button: Button
    private lateinit var tvOutput: TextView
    private val GALLERY_REQUEST_CODE = 123


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)

        imageView = binding.imageView
        button = binding.btnCaptureImage
        tvOutput = binding.tvOutput
        val buttonLoad = binding.btnLoadImage

        button.setOnClickListener{
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
            ) {
                takePicturePreview.launch(null)
            }
            else {
                requestPermission.launch(android.Manifest.permission.CAMERA)
            }
        }
        buttonLoad.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED){
                val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                intent.type = "image/*"
                val mimeTypes = arrayOf("image/jpeg","image/png","image/jpg")
                intent.putExtra(Intent.EXTRA_MIME_TYPES, mimeTypes)
                intent.flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
                onActivityResult.launch(intent)
            }else {
                requestPermission.launch(android.Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }

    }

    //request camera permission
    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()){granted->
        if (granted){
            takePicturePreview.launch(null)
        }else {
            Toast.makeText(this, "Permission Denied !! Try again", Toast.LENGTH_SHORT).show()
        }
    }

    //launch camera and take picture
    private val takePicturePreview = registerForActivityResult(ActivityResultContracts.TakePicturePreview()){bitmap->
        if (bitmap != null){
            imageView.setImageBitmap(bitmap)
            outputGenerator(bitmap = bitmap)
        }
    }

    //to get image from gallery
    private val onActivityResult= registerForActivityResult(ActivityResultContracts.StartActivityForResult()){result->
        Log.i("TAG", "This is the result: ${result.data} ${result.resultCode}")
        onResultReceived(GALLERY_REQUEST_CODE, result)
    }

    private fun onResultReceived(requestCode: Int, result: ActivityResult?){
        when(requestCode){
            GALLERY_REQUEST_CODE ->{
                if (result?.resultCode == Activity.RESULT_OK){
                    result.data?.data?.let{uri ->
                        Log.i("TAG", "onResultReceived: $uri")
                        val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
                        imageView.setImageBitmap(bitmap)
                        outputGenerator(bitmap)
                    }
                }else {
                    Log.e("TAG", "onActivityResult: error in selecting image")
                }
            }
        }
    }

    private fun outputGenerator(bitmap: Bitmap){

        //val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val model = Model.newInstance(this)

        //notes
        val newBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val tbuffer = TensorImage.fromBitmap(newBitmap)
        val byteBuffer = tbuffer.buffer

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 1), DataType.FLOAT32)
        byteBuffer.order(ByteOrder.nativeOrder())
        inputFeature0.loadBuffer(byteBuffer)
        Log.d("shape", byteBuffer.toString())
        Log.d("shape", inputFeature0.buffer.toString())

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        //val confidences = outputFeature0.floatArray

        // Releases model resources if no longer used.
        //model.close()

        //setting output text
        tvOutput.text = outputFeature0.toString()
        Log.i("TAG", "outputGenerator: $outputFeature0")

    }
}