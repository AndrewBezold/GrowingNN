package com.bezold.mnist;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/11/11
 * Time: 10:05 AM
 */
public class DigitImage {

    private int label;
    private float[] data;

    public DigitImage(int label, byte[] data) {
        this.label = label;

        this.data = new float[data.length];

        for(int i = 0; i < this.data.length; i++) {
        }
    }

    public int getLabel() {
        return label;
    }

    public float[] getData() {
        return data;
    }
}