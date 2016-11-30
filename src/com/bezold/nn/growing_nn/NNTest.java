package com.bezold.nn.growing_nn;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import no.uib.cipr.matrix.Matrix;

public class NNTest{
	
	public static void main(String[] args){
		//xor();
		growXor();
	}
	
	public static void xor(){
		GrowingNN network = new GrowingNN(2, 1000, 1);
		int epochs = 400;
		float[][] inputs = {{0,0},{0,1},{1,0},{1,1}};
		float[][] outputs = {{0},{1},{1},{0}};
		for(int i = 0; i < epochs; i++){
			for(int j = 0; j < inputs.length; j++){
				float[][] input = {inputs[j]};
				float[][] output = {outputs[j]};
				network.adam(input, output, network.learningRate, network.b1, network.b2, network.e);
			}
			float error = network.verify(inputs, outputs)[0];
			float output0 = network.feedForward(new float[][]{{0,0}}, 0, false)[0][0];
			float output1 = network.feedForward(new float[][]{{0,1}}, 0, false)[0][0];
			float output2 = network.feedForward(new float[][]{{1,0}}, 0, false)[0][0];
			float output3 = network.feedForward(new float[][]{{1,1}}, 0, false)[0][0];
			System.out.println("Epoch " + (i+1) + ": " + error);
			System.out.println("0 0: " + output0);
			System.out.println("0 1: " + output1);
			System.out.println("1 0: " + output2);
			System.out.println("1 1: " + output3);
		}
	}
	
	public static void growXor(){
		GrowingNN network = new GrowingNN(2, 3, 1);
		int epochs = 1000;
		int iterator = 0;
		int batchSize = 1;
		final Logger log = LogManager.getLogger();
		float[][] inputs = {{0,0},{0,1},{1,0},{1,1}};
		float[][] outputs = {{0},{1},{1},{0}};
		network.grow(inputs, outputs, inputs, outputs, epochs, batchSize, log);
		float output0 = network.feedForward(new float[][]{{0,0}}, 0, false)[0][0];
		float output1 = network.feedForward(new float[][]{{0,1}}, 0, false)[0][0];
		float output2 = network.feedForward(new float[][]{{1,0}}, 0, false)[0][0];
		float output3 = network.feedForward(new float[][]{{1,1}}, 0, false)[0][0];
		float error = network.verify(inputs, outputs)[0];
		System.out.println("Error: " + error);
		System.out.println("0 0: " + output0);
		System.out.println("0 1: " + output1);
		System.out.println("1 0: " + output2);
		System.out.println("1 1: " + output3);
		System.out.println(network.inputSize);
		for(int i = 0; i < network.hiddenSize.length; i++){
			System.out.println(network.hiddenSize[i]);
		}
		System.out.println(network.outputSize);
	}
	
    public static <T> void shuffle(T[] arr) {
        for (int i = arr.length - 1; i > 0; i--) {
            swap(arr, i, (int) (Math.random() * (i+1)));
        }
    }

    public static <T> void swap(T[] arr, int i, int j) {
        T tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
