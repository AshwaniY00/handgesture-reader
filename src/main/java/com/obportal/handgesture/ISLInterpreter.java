package com.obportal.handgesture;
import org.opencv.core.Core;
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

public class ISLInterpreter {
    static {
    // Load the OpenCV library that matches your system installation
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

    // Ensure checkpoints directory exists
    new java.io.File("checkpoints").mkdirs();
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
        if (!camera.isOpened()) {
            System.out.println("🚫 Failed to open camera index: " + cameraIndex);
            return;
        }

        System.out.println("🎥 Using camera index: " + cameraIndex);
        HighGui.namedWindow("ISL Interpreter", HighGui.WINDOW_NORMAL);
        HighGui.resizeWindow("ISL Interpreter", 800, 600);

        Mat frame = new Mat();
        RequestConfig config = RequestConfig.custom().setConnectTimeout(2000).setSocketTimeout(2000).build();
        CloseableHttpClient client = HttpClients.custom().setDefaultRequestConfig(config).build();

        String lastGesture = "";
        int stableCount = 0;
        StringBuilder sentence = new StringBuilder();
        long lastSent = System.currentTimeMillis();

        while (true) {
            long startTime = System.currentTimeMillis();
            camera.read(frame);
            if (frame.empty()) {
                System.out.println("⚠️ Frame is empty — skipping.");
                continue;
            }

            Mat roi = frame.clone();
            // ⏱️ Send frame every 500ms
            if (!roi.empty() && System.currentTimeMillis() - lastSent > 500) {
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

                    // 🔍 Print raw server response for debugging
                    System.out.println("🔍 Full server response: " + json);

                    String gesture = obj.optString("gesture", "");
                    double confidence = obj.optDouble("confidence", -1);
                    JSONArray box = obj.optJSONArray("box");

                    System.out.println("🧾 Gesture: " + gesture);
                    System.out.println("🧾 Confidence: " + confidence);
                    System.out.println("🧾 Sentence so far: " + sentence.toString());

                    // ✅ Only draw bounding box when server returns coordinates
                    if (box != null && box.length() == 4) {
                        int x1 = box.getInt(0), y1 = box.getInt(1), x2 = box.getInt(2), y2 = box.getInt(3);

                        // Draw the box
                        Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0), 2);

                        // Show gesture only if confidence is high
                        if (!gesture.isEmpty() && confidence > 0.07) {
                            Imgproc.putText(frame, gesture + " (" + String.format("%.2f", confidence) + ")",
                                    new Point(x1 + 10, Math.max(y1 - 10, 20)),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 255, 0), 2);
                        } else {
                            gesture = ""; // ignore low-confidence predictions
                        }
                    } else {
                        gesture = ""; // no box returned → no detection
                    }

                    // ✅ Stable detection logic (10 frames)
                    if (gesture.equals(lastGesture)) {
                        stableCount++;
                        if (stableCount >= 5 && !gesture.isEmpty()) {
                            sentence.append(gesture);
                            stableCount = 0;
                        }
                    } else {
                        lastGesture = gesture;
                        stableCount = 0;
                    }

                    // ✅ Display top predictions
                    JSONArray top = obj.optJSONArray("top_predictions");
                    if (top != null) {
                        for (int i = 0; i < top.length(); i++) {
                            JSONArray pair = top.getJSONArray(i);
                            String label = pair.getString(0);
                            double conf = pair.getDouble(1);
                            Imgproc.putText(frame, label + ": " + String.format("%.2f", conf),
                                    new Point(30, 120 + i * 30),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 200, 200), 2);
                        }
                    }

                } catch (Exception e) {
                    System.out.println("❌ Error sending image to server: " + e.getMessage());
                    e.printStackTrace();
                    Imgproc.putText(frame, "Server error", new Point(30, 90),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 0, 255), 2);
                }

                lastSent = System.currentTimeMillis();
            }

            Imgproc.putText(frame, "Gesture: " + lastGesture, new Point(30, 30),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);
            Imgproc.putText(frame, "Sentence: " + sentence.toString(), new Point(30, 60),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(200, 200, 200), 2);

            double fps = 1000.0 / (System.currentTimeMillis() - startTime);
            Imgproc.putText(frame, String.format("FPS: %.2f", fps), new Point(30, 90),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(200, 200, 200), 1);

            HighGui.imshow("ISL Interpreter", frame);
            int key = HighGui.waitKey(30);
            if (key == 27) break;
            if (key == 'r') sentence.setLength(0);
            if (key == 's') {
                try (PrintWriter out = new PrintWriter("output.txt")) {
                    out.println("Sentence: " + sentence.toString());
                    out.println("Last Gesture: " + lastGesture);
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
