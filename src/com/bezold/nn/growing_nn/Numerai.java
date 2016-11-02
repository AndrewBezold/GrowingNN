package com.bezold.nn.growing_nn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import no.uib.cipr.matrix.Matrix;

public class Numerai{
	
	public static void main(String[] args){
		numerai();
	}
	
	public static void numerai(){
		GrowingNN network = new GrowingNN(21, new int[]{50, 50, 50}, 1);
		int epochs = 100;
		float[][] inputs = new float[96320][21];
		float[][] outputs = new float[96320][1];
		
		try{
			String filename = "/home/beez/Documents/numerai_training_data.csv";
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			reader.readLine();
			for(int i = 0; i < inputs.length; i++){
				String line = reader.readLine();
				String[] data = line.split(",");
				float[] numbers = new float[data.length-1];
				for(int j = 0; j < numbers.length; j++){
					numbers[j] = Float.parseFloat(data[j]);
				}
				inputs[i] = numbers;
				outputs[i][0] = Float.parseFloat(data[data.length-1]);
			}
			reader.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		
		for(int i = 0; i < epochs; i++){
			System.out.println("Epoch: " + i);
			for(int j = 0; j < inputs.length; j++){
				float[][] input = {inputs[j]};
				float[][] output = {outputs[j]};
				network.adam(input, output, network.learningRate, network.b1, network.b2, network.e);
			}
		}
		float error = network.verify(inputs, outputs)[0];
		System.out.println("Accuracy: " + error);
	}
	
	public static void growXor(){
		GrowingNN network = new GrowingNN(2, 3, 1);
		int epochs = 400;
		int iterator = 0;
		int batchSize = 1;
		int lastIterator = 0;
		boolean endEpoch = false;
		float[][] inputs = {{0,0},{0,1},{1,0},{1,1}};
		float[][] outputs = {{0},{1},{1},{0}};
		Integer[] shuffled = {0,1,2,3};
		int newLayerNeurons = 0;
		float currentError = 0;
		float startError = network.verify(inputs, outputs)[0];
		while(iterator/inputs.length<epochs){
			if(endEpoch){
				shuffle(shuffled);
				startError = network.verify(inputs, outputs)[0];
				endEpoch = false;
				if(currentError != 0 && iterator-lastIterator>40){
					lastIterator = iterator;
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
			}
			
			for(int j = 0; j < inputs.length; j++){
				float[][] input = {inputs[shuffled[j]]};
				float[][] output = {outputs[shuffled[j]]};
				network.adam(input, output, network.learningRate, network.b1, network.b2, network.e);
				iterator++;
			}
			endEpoch = true;
			currentError = network.verify(inputs, outputs)[0];
			
			float output0 = network.feedForward(new float[][]{{0,0}}, 0, false)[0][0];
			float output1 = network.feedForward(new float[][]{{0,1}}, 0, false)[0][0];
			float output2 = network.feedForward(new float[][]{{1,0}}, 0, false)[0][0];
			float output3 = network.feedForward(new float[][]{{1,1}}, 0, false)[0][0];
			String networkSize = "";
			networkSize += network.inputSize + " ";
			for(int j = 0; j < network.hiddenSize.length; j++){
				networkSize += network.hiddenSize[j] + " ";
			}
			networkSize += network.outputSize;
			System.out.println("Epoch " + (iterator/inputs.length+1) + ": " + networkSize);
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
