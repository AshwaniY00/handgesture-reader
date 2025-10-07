package com.obportal.handgesture;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.highgui.HighGui;

import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.HttpResponse;
import org.apache.http.util.EntityUtils;
import org.json.JSONObject;

import java.io.File;

public class ISLInterpreter {
    static {
        System.load(System.getProperty("user.dir") + "/lib/libopencv_java4120.so");
    }

    public static void main(String[] args) throws Exception {
        System.out.println("🔍 Scanning for available cameras...");
        int workingIndex = -1;

        for (int i = 0; i < 5; i++) {
            VideoCapture cam = new VideoCapture(i, Videoio.CAP_V4L2);
            if (cam.isOpened()) {
                System.out.println("✅ Camera index " + i + " is available.");
                workingIndex = i;
                cam.release();
                break;
            } else {
                System.out.println("❌ No camera at index: " + i);
            }
        }

        if (workingIndex == -1) {
            System.out.println("🚫 Camera not found!");
            return;
        }

        VideoCapture camera = new VideoCapture(workingIndex);
        System.out.println("🎥 Using camera index: " + workingIndex);

        Mat frame = new Mat();
        CloseableHttpClient client = HttpClients.createDefault();
        String predictedSign = "";
        String lastGesture = "";
        int stableCount = 0;
        StringBuilder sentence = new StringBuilder();

        while (true) {
    camera.read(frame);
    if (frame.empty()) break;

    Mat roi = frame.clone(); // Send full frame to Flask

    if (!roi.empty()) {
        boolean saved = Imgcodecs.imwrite("temp.png", roi);
        if (!saved) continue;

        try {
            File imageFile = new File("temp.png");
            HttpPost post = new HttpPost("http://localhost:5000/predict");
            MultipartEntityBuilder builder = MultipartEntityBuilder.create();
            builder.addBinaryBody("image", imageFile);
            post.setEntity(builder.build());

            HttpResponse response = client.execute(post);
            String json = EntityUtils.toString(response.getEntity());
            JSONObject obj = new JSONObject(json);

            if (obj.has("gesture")) {
                predictedSign = obj.getString("gesture");

                if (predictedSign.equals(lastGesture)) {
                    stableCount++;
                    if (stableCount == 10) {
                        sentence.append(predictedSign);
                        stableCount = 0;
                    }
                } else {
                    lastGesture = predictedSign;
                    stableCount = 0;
                }

                System.out.println("🖐️ Predicted gesture: " + predictedSign);
            }

            // ✅ Draw dynamic box from Flask
            if (obj.has("box")) {
                org.json.JSONArray box = obj.getJSONArray("box");
                
                if (box.length() == 4) {
                    int x1 = box.getInt(0);
                    int y1 = box.getInt(1);
                    int x2 = box.getInt(2);
                    int y2 = box.getInt(3);
                    Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0), 2);
                }
            }

        } catch (Exception e) {
            System.out.println("❌ Error sending image to server: " + e.getMessage());
        }
    }

    Imgproc.putText(frame, "Gesture: " + predictedSign, new Point(30, 30),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);

    Imgproc.putText(frame, "Sentence: " + sentence.toString(), new Point(30, 60),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 255), 2);

   HighGui.imshow("ISL Interpreter", frame);

    int key = HighGui.waitKey(1);
    if (key == 27) break;
    if (key == 'r') sentence.setLength(0);
}


        camera.release();
        client.close();
        HighGui.destroyAllWindows();
    }
}
