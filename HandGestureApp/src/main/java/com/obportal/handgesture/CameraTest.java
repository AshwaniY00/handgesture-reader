package com.obportal.handgesture;

import org.opencv.videoio.Videoio;

import org.opencv.core.Core;
import org.opencv.videoio.VideoCapture;

public class CameraTest {
    static {
        System.load("/usr/local/lib/libopencv_java4120.so");

    }

    public static void main(String[] args) {
        for (int i = -1; i <= 3; i++) {
            VideoCapture cam = new VideoCapture(i, Videoio.CAP_V4L2);

            if (cam.isOpened()) {
                System.out.println("✅ Camera found at index: " + i);
                cam.release();
            } else {
                System.out.println("❌ No camera at index: " + i);
            }
        }
    }
}
