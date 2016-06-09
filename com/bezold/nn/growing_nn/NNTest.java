package com.bezold.nn.growing_nn;

import no.uib.cipr.matrix.Matrix;

public class NNTest{
	
	public static void main(String[] args){
		xor();
		growXor();
	}
	
	public static void xor(){
		GrowingNN network = new GrowingNN(2, 1000, 1);
		int epochs = 20;
		double[][] inputs = {{0,0},{0,1},{1,0},{1,1}};
		double[][] outputs = {{0},{1},{1},{0}};
		for(int i = 0; i < epochs; i++){
			for(int j = 0; j < inputs.length; j++){
				double[][] input = {inputs[j]};
				double[][] output = {outputs[j]};
				network.adam(input, output, network.learningRate, network.b1, network.b2, network.e);
			}
			double error = network.verify(inputs, outputs);
			double output0 = network.feedForward(new double[][]{{0,0}})[0][0];
			double output1 = network.feedForward(new double[][]{{0,1}})[0][0];
			double output2 = network.feedForward(new double[][]{{1,0}})[0][0];
			double output3 = network.feedForward(new double[][]{{1,1}})[0][0];
			System.out.println("Epoch " + (i+1) + ": " + error);
			System.out.println("0 0: " + output0);
			System.out.println("0 1: " + output1);
			System.out.println("1 0: " + output2);
			System.out.println("1 1: " + output3);
		}
	}
	
	public static void growXor(){
		GrowingNN network = new GrowingNN(2, 3, 1);
		int epochs = 20;
		double[][] inputs = {{0,0},{0,1},{1,0},{1,1}};
		double[][] outputs = {{0},{1},{1},{0}};
		Integer[] shuffled = {0,1,2,3};
		int newLayerNeurons = 0;
		double currentError = 0;
		for(int i = 0; i < epochs; i++){
			shuffle(shuffled);
			double startError = network.verify(inputs, outputs);
			if(currentError != 0){
				int numNeurons = (int) (500*(1/(1+Math.exp(0.5*((currentError-startError)-5))))*(1/(1+Math.exp(-0.5*((currentError)+5)))));
				boolean keepAdding = true;
				int layer = 2;
				while(keepAdding){
					System.out.println("Keep Adding " + (layer-2));
					if(layer < network.hiddenSize.length+2){
						System.out.println("if");
						network.addNeuron(layer, numNeurons);
						layer++;
					}else{
						System.out.println("else");
						newLayerNeurons += numNeurons;
						if(newLayerNeurons >= network.outputSize){
							System.out.println("if2");
							network.addLayer();
							newLayerNeurons = 0;
						}
						keepAdding = false;
					}
					numNeurons /= 5;
				}
			}
			for(int j = 0; j < inputs.length; j++){
				double[][] input = {inputs[shuffled[j]]};
				double[][] output = {outputs[shuffled[j]]};
				network.adam(input, output, network.learningRate, network.b1, network.b2, network.e);
			}
			currentError = network.verify(inputs, outputs);
			
			double output0 = network.feedForward(new double[][]{{0,0}})[0][0];
			double output1 = network.feedForward(new double[][]{{0,1}})[0][0];
			double output2 = network.feedForward(new double[][]{{1,0}})[0][0];
			double output3 = network.feedForward(new double[][]{{1,1}})[0][0];
			String networkSize = "";
			networkSize += network.inputSize + " ";
			for(int j = 0; j < network.hiddenSize.length; j++){
				networkSize += network.hiddenSize[j] + " ";
			}
			networkSize += network.outputSize;
			System.out.println("Epoch " + (i+1) + ": " + networkSize);
			System.out.println("Error: " + currentError);
			System.out.println("0 0: " + output0);
			System.out.println("0 1: " + output1);
			System.out.println("1 0: " + output2);
			System.out.println("1 1: " + output3);
		}
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
