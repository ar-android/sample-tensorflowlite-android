package com.ahmadrosid.sampletensorflowlite.camera


import android.app.Activity
import android.graphics.Bitmap
import android.os.SystemClock
import android.support.v4.app.Fragment
import android.support.v4.app.FragmentActivity
import android.util.Log
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.AbstractMap
import java.util.ArrayList
import java.util.Comparator
import java.util.PriorityQueue
import org.tensorflow.lite.Interpreter

/** Classifies images with Tensorflow Lite.  */
class ImageClassifier
/** Initializes an `ImageClassifier`.  */
@Throws(IOException::class)
internal constructor(activity: FragmentActivity) {


    /* Preallocated buffers for storing image data in. */
    private val intValues = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    private var tflite: Interpreter? = null

    /** Labels corresponding to the output of the vision model.  */
    private val labelList: List<String>

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.  */
    private var imgData: ByteBuffer? = null

    /** An array to hold inference results, to be feed into Tensorflow Lite as outputs.  */
    private var labelProbArray: Array<FloatArray>? = null
    /** multi-stage low pass filter  */
    private var filterLabelProbArray: Array<FloatArray>? = null

    private val sortedLabels = PriorityQueue<Map.Entry<String, Float>>(
        RESULTS_TO_SHOW,
        Comparator<Map.Entry<String, Float>> { o1, o2 -> o1.value.compareTo(o2.value) })

    init {
        tflite = Interpreter(loadModelFile(activity))
        labelList = loadLabelList(activity)
        imgData = ByteBuffer.allocateDirect(
            4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE
        )
        imgData!!.order(ByteOrder.nativeOrder())
        labelProbArray = Array(1) { FloatArray(labelList.size) }
        filterLabelProbArray = Array(FILTER_STAGES) { FloatArray(labelList.size) }
        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.")
    }

    /** Classifies a frame from the preview stream.  */
    internal fun classifyFrame(bitmap: Bitmap): String {
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.")
            return "Uninitialized Classifier."
        }
        convertBitmapToByteBuffer(bitmap)
        // Here's where the magic happens!!!
        val startTime = SystemClock.uptimeMillis()
        tflite!!.run(imgData!!, labelProbArray!!)
        val endTime = SystemClock.uptimeMillis()
        Log.d(TAG, "Timecost to run model inference: " + java.lang.Long.toString(endTime - startTime))

        // smooth the results
        applyFilter()

        // print the results
        var textToShow = printTopKLabels()
        textToShow = java.lang.Long.toString(endTime - startTime) + "ms" + textToShow
        return textToShow
    }

    internal fun applyFilter() {
        val num_labels = labelList.size

        // Low pass filter `labelProbArray` into the first stage of the filter.
        for (j in 0 until num_labels) {
            filterLabelProbArray!![0][j] += FILTER_FACTOR * (labelProbArray!![0][j] - filterLabelProbArray!![0][j])
        }
        // Low pass filter each stage into the next.
        for (i in 1 until FILTER_STAGES) {
            for (j in 0 until num_labels) {
                filterLabelProbArray!![i][j] += FILTER_FACTOR * (filterLabelProbArray!![i - 1][j] - filterLabelProbArray!![i][j])

            }
        }

        // Copy the last stage filter output back to `labelProbArray`.
        for (j in 0 until num_labels) {
            labelProbArray!![0][j] = filterLabelProbArray!![FILTER_STAGES - 1][j]
        }
    }

    /** Closes tflite to release resources.  */
    fun close() {
        tflite!!.close()
        tflite = null
    }

    /** Reads label list from Assets.  */
    @Throws(IOException::class)
    private fun loadLabelList(activity: Activity): List<String> {
        val labelList = ArrayList<String>()
        val reader = BufferedReader(InputStreamReader(activity.assets.open(LABEL_PATH)))
        reader.lineSequence().forEach {
            labelList.add(it)
        }
        reader.close()
        return labelList
    }

    /** Memory-map the model file in Assets.  */
    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /** Writes Image data into a `ByteBuffer`.  */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        if (imgData == null) {
            return
        }
        imgData!!.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        // Convert the image to floating point.
        var pixel = 0
        val startTime = SystemClock.uptimeMillis()
        for (i in 0 until DIM_IMG_SIZE_X) {
            for (j in 0 until DIM_IMG_SIZE_Y) {
                val `val` = intValues[pixel++]
                imgData!!.putFloat(((`val` shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                imgData!!.putFloat(((`val` shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                imgData!!.putFloat(((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
        val endTime = SystemClock.uptimeMillis()
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + java.lang.Long.toString(endTime - startTime))
    }

    /** Prints top-K labels, to be shown in UI as the results.  */
    private fun printTopKLabels(): String {
        for (i in labelList.indices) {
            sortedLabels.add(
                AbstractMap.SimpleEntry(labelList[i], labelProbArray!![0][i])
            )
            if (sortedLabels.size > RESULTS_TO_SHOW) {
                sortedLabels.poll()
            }
        }
        var textToShow = ""
        val size = sortedLabels.size
        for (i in 0 until size) {
            val label = sortedLabels.poll()
            textToShow = String.format("\n%s: %4.2f", label.key, label.value) + textToShow
        }
        return textToShow
    }

    companion object {

        /** Tag for the [Log].  */
        private val TAG = "TfLiteCameraDemo"

        /** Name of the model file stored in Assets.  */
        private val MODEL_PATH = "graph.lite"

        /** Name of the label file stored in Assets.  */
        private val LABEL_PATH = "labels.txt"

        /** Number of results to show in the UI.  */
        private val RESULTS_TO_SHOW = 3

        /** Dimensions of inputs.  */
        private val DIM_BATCH_SIZE = 1

        private val DIM_PIXEL_SIZE = 3

        internal val DIM_IMG_SIZE_X = 224
        internal val DIM_IMG_SIZE_Y = 224

        private val IMAGE_MEAN = 128
        private val IMAGE_STD = 128.0f
        private val FILTER_STAGES = 3
        private val FILTER_FACTOR = 0.4f
    }
}