package com.obportal.handgesture;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.highgui.HighGui;

import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.HttpResponse;
import org.apache.http.util.EntityUtils;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.PrintWriter;
import java.util.Arrays;

public class ISLInterpreter {
    static {
        System.load(System.getProperty("user.dir") + "/lib/libopencv_java4120.so");
        new java.io.File("checkpoints").mkdirs(); // ✅ Ensure checkpoints folder exists
    }

    public static void main(String[] args) throws Exception {
        System.out.println("🔍 Scanning for available cameras...");
        int cameraIndex = -1;

        for (int i = 0; i < 5; i++) {
            VideoCapture testCam = new VideoCapture(i, Videoio.CAP_V4L2);
            if (testCam.isOpened()) {
                System.out.println("✅ Camera index " + i + " is available.");
                cameraIndex = i;
                testCam.release();
                break;
            } else {
                System.out.println("❌ No camera at index: " + i);
            }
        }

        if (cameraIndex == -1) {
            System.out.println("🚫 Camera not found!");
            return;
        }

        VideoCapture camera = new VideoCapture(cameraIndex);
        System.out.println("🎥 Using camera index: " + cameraIndex);
        HighGui.namedWindow("ISL Interpreter", HighGui.WINDOW_NORMAL);
        HighGui.resizeWindow("ISL Interpreter", 800, 600);

        Mat frame = new Mat();
        RequestConfig config = RequestConfig.custom()
                .setConnectTimeout(2000)
                .setSocketTimeout(2000)
                .build();
        CloseableHttpClient client = HttpClients.custom().setDefaultRequestConfig(config).build();

        String lastGesture = "";
        int stableCount = 0;
        StringBuilder sentence = new StringBuilder();
        long lastSent = System.currentTimeMillis();

        int cropX1 = -1, cropY1 = -1, cropX2 = -1, cropY2 = -1;

        while (true) {
            long startTime = System.currentTimeMillis();

            camera.read(frame);
            if (frame.empty()) break;

            Mat roi = frame.clone(); // ✅ Send full frame

            if (!roi.empty() && System.currentTimeMillis() - lastSent > 100) { // ⚡ Faster prediction loop
                MatOfByte mob = new MatOfByte();
                Imgcodecs.imencode(".png", roi, mob);
                byte[] imageBytes = mob.toArray();

                try {
                    HttpPost post = new HttpPost("http://localhost:5001/predict");
                    MultipartEntityBuilder builder = MultipartEntityBuilder.create();
                    builder.addBinaryBody("image", imageBytes, ContentType.DEFAULT_BINARY, "frame.png");
                    post.setEntity(builder.build());

                    HttpResponse response = client.execute(post);
                    String json = EntityUtils.toString(response.getEntity());
                    JSONObject obj = new JSONObject(json);

                    String gesture = obj.optString("gesture", "");
                    JSONArray box = obj.optJSONArray("box");

                    if (gesture.equals(lastGesture)) {
                        stableCount++;
                        if (stableCount == 10 && !gesture.isEmpty()) {
                            sentence.append(gesture);
                            stableCount = 0;
                        }
                    } else {
                        lastGesture = gesture;
                        stableCount = 0;
                    }

                    if (box != null && box.length() == 4) {
                        cropX1 = box.getInt(0);
                        cropY1 = box.getInt(1);
                        cropX2 = box.getInt(2);
                        cropY2 = box.getInt(3);

                        Imgproc.rectangle(frame, new Point(cropX1, cropY1), new Point(cropX2, cropY2), new Scalar(0, 255, 0), 2);
                        Imgproc.putText(frame, gesture, new Point(cropX1 + 10, Math.max(cropY1 - 10, 20)),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 255, 0), 2);
                    } else {
                        cropX1 = cropY1 = cropX2 = cropY2 = -1;
                    }

                } catch (Exception e) {
                    System.out.println("❌ Error sending image to server: " + e.getMessage());
                    Imgproc.putText(frame, "Server error", new Point(30, 90),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 0, 255), 2);
                }

                lastSent = System.currentTimeMillis();
            }

            Imgproc.putText(frame, "Gesture: " + lastGesture, new Point(30, 30),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);
            Imgproc.putText(frame, "Sentence: " + sentence.toString(), new Point(30, 60),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 255), 2); // ✅ Live sentence display

            long endTime = System.currentTimeMillis();
            double fps = 1000.0 / (endTime - startTime);
            Imgproc.putText(frame, String.format("FPS: %.2f", fps), new Point(30, 90),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(200, 200, 200), 1);

            HighGui.imshow("ISL Interpreter", frame);
            int key = HighGui.waitKey(1);

            if (key == 27) break;
            if (key == 'r') sentence.setLength(0);
            if (key == 's') {
                try (PrintWriter out = new PrintWriter("output.txt")) {
                    out.println(sentence.toString());
                    System.out.println("💾 Sentence saved to output.txt");
                    Thread.sleep(500);
                } catch (Exception e) {
                    System.out.println("❌ Failed to save sentence: " + e.getMessage());
                }
            }
        }

        camera.release();
        client.close();
        HighGui.destroyAllWindows();
    }
}
