package com.bezold.mnist;

/**
 * Created by IntelliJ IDEA.
 * User: vivin
 * Date: 11/11/11
 * Time: 10:05 AM
 */
public class DigitImage {

    private int label;
    private double[] data;

    public DigitImage(int label, byte[] data) {
        this.label = label;

        this.data = new double[data.length];

        for(int i = 0; i < this.data.length; i++) {
        }
    }

    public int getLabel() {
        return label;
    }

    public double[] getData() {
        return data;
    }
}