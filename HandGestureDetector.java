import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.util.ArrayList;
import java.util.List;

public class HandGestureDetector {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // Load OpenCV native lib
    }

    public static void main(String[] args) {
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Error: Camera not found!");
            return;
        }

        Mat frame = new Mat();
        Mat gray = new Mat();
        Mat blurred = new Mat();
        Mat thresh = new Mat();

        while (true) {
            camera.read(frame);
            if (frame.empty()) break;

            // Preprocessing
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(gray, blurred, new Size(35, 35), 0);
            Imgproc.threshold(blurred, thresh, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            // Find contours
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

            if (!contours.isEmpty()) {
                double maxArea = 0;
                int maxIndex = 0;
                for (int i = 0; i < contours.size(); i++) {
                    double area = Imgproc.contourArea(contours.get(i));
                    if (area > maxArea) {
                        maxArea = area;
                        maxIndex = i;
                    }
                }

                MatOfPoint handContour = contours.get(maxIndex);
                Rect boundingBox = Imgproc.boundingRect(handContour);
                Imgproc.rectangle(frame, boundingBox, new Scalar(0, 255, 0), 2);

                String gesture = (maxArea > 5000) ? "Open Palm" : "Fist";
                Imgproc.putText(frame, "Gesture: " + gesture, new Point(30, 30),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);
            }

            HighGui.imshow("Hand Gesture Detector", frame);
            if (HighGui.waitKey(1) == 27) break; // ESC to exit
        }

        camera.release();
        HighGui.destroyAllWindows();
    }
}
