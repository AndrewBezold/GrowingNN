package com.bezold.nn.growing_nn;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class GrowingNN implements Cloneable{
	
	int inputSize;
	int[] hiddenSize;
	int outputSize;
	
	ArrayList<Matrix> matrices;
	ArrayList<Matrix> biases;
	
	Matrix inputLayer;
	ArrayList<Matrix> hiddenLayer;
	Matrix outputLayer;
	
	ArrayList<Matrix> m;
	ArrayList<Matrix> v;
	int t;
	
	double learningRate = .001;
	double b1 = .9;
	double b2 = .999;
	double e = .0001;
	
	public GrowingNN(int inputSize, int hiddenSize, int outputSize){
		this(inputSize, new int[]{hiddenSize}, outputSize);
	}
	
	public GrowingNN(int inputSize, int[] hiddenSize, int outputSize){
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.outputSize = outputSize;
		
		matrices = new ArrayList<Matrix>(hiddenSize.length+1);
		for(int i = 0; i < hiddenSize.length+1; i++){
			if(i==0){
				matrices.add(initWeights(inputSize, hiddenSize[0]));
			}else if(i==hiddenSize.length){
				matrices.add(initWeights(hiddenSize[i-1], outputSize));
			}else{
				matrices.add(initWeights(hiddenSize[i-1], hiddenSize[i]));
			}
		}
		
		biases = new ArrayList<Matrix>(hiddenSize.length+1);
		for(int i = 0; i < hiddenSize.length+1; i++){
			if(i==hiddenSize.length){
				biases.add(initWeights(1, outputSize));
			}else{
				biases.add(initWeights(1, hiddenSize[i]));
			}
		}
		
		m = new ArrayList<Matrix>(matrices.size() + biases.size());
		v = new ArrayList<Matrix>(matrices.size() + biases.size());
		int count = 0;
		for(int i = 0; i < matrices.size()+biases.size(); i++){
			if(i%2==0){
				m.add(new DenseMatrix(matrices.get(count).numRows(), matrices.get(count).numColumns()).zero());
				v.add(new DenseMatrix(matrices.get(count).numRows(), matrices.get(count).numColumns()).zero());
			}else{
				m.add(new DenseMatrix(biases.get(count).numRows(), biases.get(count).numColumns()).zero());
				v.add(new DenseMatrix(biases.get(count).numRows(), biases.get(count).numColumns()).zero());
				count++;
			}
		}
		t = 0;
	}
	
	public GrowingNN(GrowingNN network){
		inputSize = network.inputSize;
		hiddenSize = new int[network.hiddenSize.length];
		for(int i = 0; i < network.hiddenSize.length; i++){
			hiddenSize[i] = network.hiddenSize[i];
		}
		outputSize = network.outputSize;
		matrices = new ArrayList<Matrix>(network.matrices.size());
		for(int i = 0; i < network.matrices.size(); i++){
			double[][] matrixArray = new double[network.matrices.get(i).numRows()][network.matrices.get(i).numColumns()];
			for(int j = 0; j < network.matrices.get(i).numRows(); j++){
				for(int k = 0; k < network.matrices.get(i).numColumns(); k++){
					matrixArray[j][k] = network.matrices.get(i).get(j, k);
				}
			}
			matrices.add(new DenseMatrix(matrixArray));
		}
		biases = new ArrayList<Matrix>(network.biases.size());
		for(int i = 0; i < network.biases.size(); i++){
			double[][] biasArray = new double[network.biases.get(i).numRows()][network.biases.get(i).numColumns()];
			for(int j = 0; j < network.biases.get(i).numRows(); j++){
				for(int k = 0; k < network.biases.get(i).numColumns(); k++){
					biasArray[j][k] = network.biases.get(i).get(j, k);
				}
			}
			biases.add(new DenseMatrix(biasArray));
		}
		double[][] inputArray = new double[network.inputLayer.numRows()][network.inputLayer.numColumns()];
		for(int j = 0; j < network.inputLayer.numRows(); j++){
			for(int k = 0; k < network.inputLayer.numColumns(); k++){
				inputArray[j][k] = network.inputLayer.get(j, k);
			}
		}
		inputLayer = new DenseMatrix(inputArray);
		hiddenLayer = new ArrayList<Matrix>(network.hiddenLayer.size());
		for(int i = 0; i < network.hiddenLayer.size(); i++){
			double[][] hiddenArray = new double[network.hiddenLayer.get(i).numRows()][network.hiddenLayer.get(i).numColumns()];
			for(int j = 0; j < network.hiddenLayer.get(i).numRows(); j++){
				for(int k = 0; k < network.hiddenLayer.get(i).numColumns(); k++){
					hiddenArray[j][k] = network.hiddenLayer.get(i).get(j, k);
				}
			}
			hiddenLayer.add(new DenseMatrix(hiddenArray));
		}
		double[][] outputArray = new double[network.outputLayer.numRows()][network.outputLayer.numColumns()];
		for(int j = 0; j < network.outputLayer.numRows(); j++){
			for(int k = 0; k < network.outputLayer.numColumns(); k++){
				outputArray[j][k] = network.outputLayer.get(j, k);
			}
		}
		outputLayer = new DenseMatrix(outputArray);
		m = new ArrayList<Matrix>(network.m.size());
		for(int i = 0; i < network.m.size(); i++){
			double[][] mArray = new double[network.m.get(i).numRows()][network.m.get(i).numColumns()];
			for(int j = 0; j < network.m.get(i).numRows(); j++){
				for(int k = 0; k < network.m.get(i).numColumns(); k++){
					mArray[j][k] = network.m.get(i).get(j, k);
				}
			}
			m.add(new DenseMatrix(mArray));
		}
		v = new ArrayList<Matrix>(network.v.size());
		for(int i = 0; i < network.v.size(); i++){
			double[][] vArray = new double[network.v.get(i).numRows()][network.v.get(i).numColumns()];
			for(int j = 0; j < network.v.get(i).numRows(); j++){
				for(int k = 0; k < network.v.get(i).numColumns(); k++){
					vArray[j][k] = network.v.get(i).get(j, k);
				}
			}
			v.add(new DenseMatrix(vArray));
		}
		
		t = network.t;
		learningRate = network.learningRate;
		b1 = network.b1;
		b2 = network.b2;
		e = network.e;
		
	}
	
	public double[][] feedForward(double[][] inputs){
		inputLayer = new DenseMatrix(inputs);
		outputLayer = feedForward(inputLayer);
		double[][] outputs = new double[outputLayer.numRows()][outputLayer.numColumns()];
		for(int i = 0; i < outputs.length; i++){
			for(int j = 0; j < outputs[i].length; j++){
				outputs[i][j] = outputLayer.get(i, j);
			}
		}
		return outputs;
	}
	
	public Matrix feedForward(Matrix inputs){
		inputLayer = inputs;
		hiddenLayer = new ArrayList<Matrix>(hiddenSize.length);
		ArrayList<Matrix> biases = resizeBiases(this.biases, inputs);
		for(int i = 0; i < hiddenSize.length; i++){
			if(i==0){
				hiddenLayer.add(tanh(inputLayer.multAdd(matrices.get(i), biases.get(i).copy())));
			}else{
				hiddenLayer.add(tanh(hiddenLayer.get(i-1).multAdd(matrices.get(i), biases.get(i).copy())));
			}
		}
		outputLayer = tanh(hiddenLayer.get(hiddenLayer.size()-1).multAdd(matrices.get(matrices.size()-1), biases.get(biases.size()-1).copy()));
		return outputLayer;
	}
	
	public Matrix[] gradients(Matrix inputs, Matrix expectedOutput){
		//make gradients of biases
		Matrix output = feedForward(inputs);
		Matrix error = output.copy().add(-1, expectedOutput);
		//change weight by derivative of error
		Matrix[] delta = new Matrix[matrices.size()];
		Matrix[] deltaBias = new Matrix[matrices.size()];
		Matrix[] dW = new Matrix[matrices.size()];
		Matrix[] dWBias = new Matrix[matrices.size()];
		for(int i = matrices.size()-1; i >= 0; i--){
			if(i==matrices.size()-1){
				Matrix derivative = dTanh(output);
				delta[i] = elementMult(error, derivative);
				deltaBias[i] = elementMult(error, derivative);
			}else{
				Matrix derivative = dTanh(hiddenLayer.get(i));
				delta[i] = new DenseMatrix(delta[i+1].numRows(), matrices.get(i+1).numRows());
				delta[i] = elementMult(delta[i+1].transBmult(matrices.get(i+1).copy(), delta[i]), derivative);
				deltaBias[i] = new DenseMatrix(delta[i+1].numRows(), matrices.get(i+1).numRows());
				deltaBias[i] = elementMult(delta[i+1].transBmult(matrices.get(i+1).copy(), deltaBias[i]), derivative);
			}
			if(i!=0){
				dW[i] = new DenseMatrix(matrices.get(i).numRows(), matrices.get(i).numColumns());
				tanh(hiddenLayer.get(i-1)).transAmult(delta[i], dW[i]);
				dWBias[i] = new DenseMatrix(biases.get(i).numRows(), biases.get(i).numColumns());
				Matrix ones = new DenseMatrix(deltaBias[i].numRows(), biases.get(i).numRows());
				for(int j = 0; j < ones.numRows(); j++){
					for(int k = 0; k < ones.numColumns(); k++){
						ones.set(j, k, 1);
					}
				}
				ones.transAmult(deltaBias[i], dWBias[i]);
			}else{
				dW[i] = new DenseMatrix(matrices.get(i).numRows(), matrices.get(i).numColumns());
				tanh(inputLayer).transAmult(delta[i], dW[i]);
				dWBias[i] = new DenseMatrix(biases.get(i).numRows(), biases.get(i).numColumns());
				Matrix ones = new DenseMatrix(deltaBias[i].numRows(), biases.get(i).numRows());
				for(int j = 0; j < ones.numRows(); j++){
					for(int k = 0; k < ones.numColumns(); k++){
						ones.set(j, k, 1);
					}
				}
				ones.transAmult(deltaBias[i], dWBias[i]);
			}
		}
		double[][] averagingDouble = new double[1][error.numRows()];
		for(int i = 0; i < averagingDouble.length; i++){
			for(int j = 0; j < averagingDouble[i].length; j++){
				averagingDouble[i][j] = 1;
			}
		}
		Matrix averaging = new DenseMatrix(averagingDouble);
		Matrix avgError = new DenseMatrix(averaging.numRows(), error.numColumns());
		avgError = averaging.mult(1/(double)error.numRows(), error, avgError);
		Matrix[] returned = new Matrix[dW.length + dWBias.length + 1];
		int count = 0;
		for(int i = 0; i < returned.length; i++){
			if(i < returned.length - 1){
				returned[i] = dW[count];
				returned[i+1] = dWBias[count];
				i++;
				count++;
			}else{
				returned[i] = avgError;
			}
		}
		return returned;
	}
	
	public void backpropagate(double[][] inputs, double[][] expectedOutput, double learningRate){
		Matrix[] dW = gradients(new DenseMatrix(inputs), new DenseMatrix(expectedOutput));
		for(int i = 0; i < matrices.size(); i++){
			matrices.set(i, matrices.get(i).add(-1, dW[i].copy().scale(learningRate)));
		}
	}
	
	//speed this up
	public Matrix[] adam(double[][] inputs, double[][] expectedOutput, double learningRate, double b1, double b2, double e){
		t++;
		Matrix[] g = gradients(new DenseMatrix(inputs), new DenseMatrix(expectedOutput));
		int count = 0;
		for(int i = 0; i < matrices.size() + biases.size(); i++){
			m.get(i).scale(b1).add(g[i].copy().scale(1-b1));
			v.get(i).scale(b2).add(elementMult(g[i], g[i]).scale(1-b2));
			double alpha = learningRate * Math.sqrt(1 - Math.pow(b2, t))/(1-Math.pow(b1, t));
			if(i%2==0){
				matrices.get(count).add(-1, elementAdam(alpha, m.get(i), v.get(i), e));
			}else{
				biases.get(count).add(-1, elementAdam(alpha, m.get(i), v.get(i), e));
				count++;
			}
		}
		return g;
	}
	
	public double[] verify(double[][] inputs, int[] expectedOutput){
		double accuracy = 0;
		double absError = 0;
		double[][] output = feedForward(inputs);
		for(int i = 0; i < output.length; i++){
			int max = 0;
			for(int j = 1; j < output[i].length; j++){
				if(output[i][j] > output[i][max]){
					max = j;
				}
			}
			if(expectedOutput[i]==max){
				accuracy++;
			}
			for(int j = 0; j < output[i].length; j++){
				if(expectedOutput[i] == j){
					absError += Math.abs(1 - output[i][j]);
				}else{
					absError += Math.abs(0 - output[i][j]);
				}
			}
		}
		accuracy = accuracy / inputs.length;
		absError /= inputs.length;
		return new double[]{accuracy, absError};
	}
	
	public double verify(double[][] inputs, double[][] expectedOutput){
		double absError = 0;
		double[][] output = feedForward(inputs);
		for(int i = 0; i < output.length; i++){
			for(int j = 0; j < output[i].length; j++){
				absError += Math.abs(expectedOutput[i][j] - output[i][j]);
			}
		}
		absError /= inputs.length;
		return absError;
	}
	
	//set weights to between -0.1 and 0.1
	public Matrix initWeights(int rows, int columns){
		double[][] weights = initValues(rows, columns);
		Matrix matrix = new DenseMatrix(weights);
		return matrix;
	}
	
	public void addLayer(){
		int[] newHiddenSize = new int[hiddenSize.length+1];
		for(int i = 0; i < hiddenSize.length; i++){
			newHiddenSize[i] = hiddenSize[i];
		}
		newHiddenSize[newHiddenSize.length-1] = outputSize;
		hiddenSize = newHiddenSize;
		
		matrices.add(initWeights(outputSize, outputSize));
		
		biases.add(initWeights(1, outputSize));
		
		m.add((new DenseMatrix(outputSize, outputSize)).zero());
		m.add((new DenseMatrix(1, outputSize)).zero());
		
		v.add((new DenseMatrix(outputSize, outputSize)).zero());
		v.add((new DenseMatrix(1, outputSize)).zero());
	}
	
	//layernum = the layer that gets the new neuron.  Input is 1, output is last
	public void addNeuron(int layerNum, int numNeurons){
		hiddenSize[layerNum-2]++;
		matrices.set(layerNum-2, addColumn(matrices.get(layerNum-2), numNeurons, true));
		matrices.set(layerNum-1, addRow(matrices.get(layerNum-1), numNeurons, true));
		biases.set(layerNum-2, addColumn(biases.get(layerNum-2), numNeurons, true));
		m.set(2*(layerNum-2), addColumn(m.get(2*(layerNum-2)), numNeurons, false));
		m.set(2*(layerNum-1), addRow(m.get(2*(layerNum-1)), numNeurons, false));
		m.set(2*(layerNum-2)+1, addColumn(m.get(2*(layerNum-2)+1), numNeurons, false));
		v.set(2*(layerNum-2), addColumn(v.get(2*(layerNum-2)), numNeurons, false));
		v.set(2*(layerNum-1), addRow(v.get(2*(layerNum-1)), numNeurons, false));
		v.set(2*(layerNum-2)+1, addColumn(v.get(2*(layerNum-2)+1), numNeurons, false));
	}
	
	public Matrix addRow(Matrix matrix, int numNeurons, boolean isNetwork){
		int rows = matrix.numRows();
		int col = matrix.numColumns();
		double[][] weights = new double[rows+numNeurons][col];
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < col; j++){
				weights[i][j] = matrix.get(i, j);
			}
		}
		double[][] newRow;
		if(isNetwork){
			newRow = initValues(numNeurons, col);
		}else{
			newRow = new double[numNeurons][col];
			for(int i = 0; i < col; i++){
				for(int j = 0; j < numNeurons; j++){
					newRow[j][i] = 0;
				}
			}
		}
		for(int i = 0; i < col; i++){
			for(int j = 0; j < numNeurons; j++){
				weights[weights.length-numNeurons+j][i] = newRow[j][i];
			}
		}
		Matrix newMatrix = new DenseMatrix(weights);
		return newMatrix;
	}
	
	public Matrix addColumn(Matrix matrix, int numNeurons, boolean isNetwork){
		int rows = matrix.numRows();
		int col = matrix.numColumns();
		double[][] weights = new double[rows][col+numNeurons];
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < col; j++){
				weights[i][j] = matrix.get(i, j);
			}
		}
		double[][] newCol;
		if(isNetwork){
			newCol = initValues(rows, numNeurons);
		}else{
			newCol = new double[rows][numNeurons];
			for(int i = 0; i < rows; i++){
				for(int j = 0; j < numNeurons; j++){
					newCol[i][j] = 0;
				}
			}
		}
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < numNeurons; j++){
				weights[i][weights[i].length-numNeurons+j] = newCol[i][j];
			}
		}
		Matrix newMatrix = new DenseMatrix(weights);
		return newMatrix;
	}
	
	public double[][] initValues(int rows, int col){
		double[][] weights = new double[rows][col];
		for(int i = 0; i < weights.length; i++){
			for(int j = 0; j < weights[i].length; j++){
				weights[i][j] = Math.random() * .2 - .1;
			}
		}
		return weights;
	}
	
	public Matrix tanh(Matrix x){
		double[][] value = new double[x.numRows()][x.numColumns()];
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				value[i][j] = Math.tanh(x.get(i, j));
			}
		}
		Matrix returnMatrix = new DenseMatrix(value);
		return returnMatrix;
	}
	
	public Matrix dTanh(Matrix x){
		double[][] value = new double[x.numRows()][x.numColumns()];
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				value[i][j] = 1 - Math.pow(Math.tanh(x.get(i, j)), 2);
			}
		}
		Matrix returnMatrix = new DenseMatrix(value);
		return returnMatrix;
	}
	
	public Matrix elementMult(Matrix a, Matrix b){
		Matrix c = new DenseMatrix(a.numRows(), a.numColumns());
		for(int i = 0; i < a.numRows(); i++){
			for(int j = 0; j < a.numColumns(); j++){
				c.set(i, j, a.get(i, j) * b.get(i, j));
			}
		}
		return c;
	}
	
	public Matrix elementAdam(double alpha, Matrix m, Matrix v, double e){
		double[][] value = new double[m.numRows()][m.numColumns()];
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				value[i][j] = alpha * m.get(i, j) / (Math.sqrt(v.get(i, j)) + e);
			}
		}
		Matrix resultMatrix = new DenseMatrix(value);
		return resultMatrix;
	}
	
	public ArrayList<Matrix> resizeBiases(ArrayList<Matrix> biases, Matrix inputs){
		double[][][] resizedArray = new double[biases.size()][][];
		ArrayList<Matrix> resized = new ArrayList<Matrix>(biases.size());
		for(int n = 0; n < biases.size(); n++){
			resizedArray[n] = new double[inputs.numRows()][biases.get(n).numColumns()];
			for(int i = 0; i < resizedArray[n].length; i++){
				for(int j = 0; j < resizedArray[n][i].length; j++){
					resizedArray[n][i][j] = biases.get(n).get(0, j);
				}
			}
			resized.add(new DenseMatrix(resizedArray[n]));
		}
		return resized;
	}
	
	public void output(String fileName) throws IOException{
		File file = new File(fileName);
		FileWriter writer = new FileWriter(file);
		String output = output();
		writer.write(output);
		writer.close();
	}
	
	public String output(){
		String output = "";
		output += (hiddenSize.length + 2) + System.lineSeparator();
		output += inputSize + System.lineSeparator();
		for(int i = 0; i < hiddenSize.length; i++){
			output += hiddenSize[i] + System.lineSeparator();
		}
		output += outputSize + System.lineSeparator();
		for(int i = 0; i < matrices.size(); i++){
			for(int j = 0; j < matrices.get(i).numRows(); j++){
				for(int k = 0; k < matrices.get(i).numColumns(); k++){
					output += matrices.get(i).get(j, k);
					if(k!=matrices.get(i).numColumns() - 1){
						output += " ";
					}
				}
				output += System.lineSeparator();
			}
			System.out.println("Finished matrix " + i);
			for(int j = 0; j < biases.get(i).numRows(); j++){
				for(int k = 0; k < biases.get(i).numColumns(); k++){
					output += biases.get(i).get(j, k);
					if(k!=biases.get(i).numColumns()-1){
						output += " ";
					}
				}
				if(!(i==matrices.size()-1&&j==biases.get(i).numRows()-1)){
					output += System.lineSeparator();
				}
			}
			System.out.println("Finished bias " + i);
		}
		return output;
	}
	
	public GrowingNN clone(){
		GrowingNN clone = new GrowingNN(inputSize, hiddenSize, outputSize);
		clone.matrices = new ArrayList<Matrix>(matrices.size());
		clone.biases = new ArrayList<Matrix>(biases.size());
		for(int i = 0; i < matrices.size(); i++){
			clone.matrices.set(i, matrices.get(i).copy());
			clone.biases.set(i, biases.get(i).copy());
		}
		clone.inputLayer = inputLayer.copy();
		clone.hiddenLayer = new ArrayList<Matrix>(hiddenLayer.size());
		for(int i = 0; i < hiddenLayer.size(); i++){
			clone.hiddenLayer.set(i, hiddenLayer.get(i).copy());
		}
		clone.outputLayer = outputLayer.copy();
		clone.m = new ArrayList<Matrix>(m.size());
		clone.v = new ArrayList<Matrix>(v.size());
		for(int i = 0; i < m.size(); i++){
			clone.m.set(i, m.get(i).copy());
			clone.v.set(i, v.get(i).copy());
		}
		clone.t = t;
		clone.learningRate = learningRate;
		clone.b1 = b1;
		clone.b2 = b2;
		clone.e = e;
		return clone;
	}

}
