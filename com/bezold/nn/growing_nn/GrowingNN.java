package com.bezold.nn.growing_nn;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class GrowingNN implements Cloneable{
	
	int inputSize;
	int[] hiddenSize;
	int outputSize;
	
	Matrix[] matrices;
	Matrix[] biases;
	
	Matrix inputLayer;
	Matrix[] hiddenLayer;
	Matrix outputLayer;
	
	Matrix[] m;
	Matrix[] v;
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
		
		matrices = new Matrix[hiddenSize.length+1];
		for(int i = 0; i < matrices.length; i++){
			if(i==0){
				matrices[i] = initWeights(inputSize, hiddenSize[0]);
			}else if(i==matrices.length-1){
				matrices[i] = initWeights(hiddenSize[i-1], outputSize);
			}else{
				matrices[i] = initWeights(hiddenSize[i-1], hiddenSize[i]);
			}
		}
		
		biases = new Matrix[hiddenSize.length+1];
		for(int i = 0; i < biases.length; i++){
			if(i==biases.length-1){
				biases[i] = initWeights(1, outputSize);
			}else{
				biases[i] = initWeights(1, hiddenSize[i]);
			}
		}
		
		m = new Matrix[matrices.length + biases.length];
		v = new Matrix[matrices.length + biases.length];
		int count = 0;
		for(int i = 0; i < m.length; i++){
			if(i%2==0){
				m[i] = new DenseMatrix(matrices[count].numRows(), matrices[count].numColumns()).zero();
				v[i] = new DenseMatrix(matrices[count].numRows(), matrices[count].numColumns()).zero();
			}else{
				m[i] = new DenseMatrix(biases[count].numRows(), biases[count].numColumns()).zero();
				v[i] = new DenseMatrix(biases[count].numRows(), biases[count].numColumns()).zero();
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
		matrices = new Matrix[network.matrices.length];
		for(int i = 0; i < network.matrices.length; i++){
			double[][] matrixArray = new double[network.matrices[i].numRows()][network.matrices[i].numColumns()];
			for(int j = 0; j < network.matrices[i].numRows(); j++){
				for(int k = 0; k < network.matrices[i].numColumns(); k++){
					matrixArray[j][k] = network.matrices[i].get(j, k);
				}
			}
			matrices[i] = new DenseMatrix(matrixArray);
		}
		biases = new Matrix[network.biases.length];
		for(int i = 0; i < network.biases.length; i++){
			double[][] biasArray = new double[network.biases[i].numRows()][network.biases[i].numColumns()];
			for(int j = 0; j < network.biases[i].numRows(); j++){
				for(int k = 0; k < network.biases[i].numColumns(); k++){
					biasArray[j][k] = network.biases[i].get(j, k);
				}
			}
			biases[i] = new DenseMatrix(biasArray);
		}
		double[][] inputArray = new double[network.inputLayer.numRows()][network.inputLayer.numColumns()];
		for(int j = 0; j < network.inputLayer.numRows(); j++){
			for(int k = 0; k < network.inputLayer.numColumns(); k++){
				inputArray[j][k] = network.inputLayer.get(j, k);
			}
		}
		inputLayer = new DenseMatrix(inputArray);
		hiddenLayer = new Matrix[network.hiddenLayer.length];
		for(int i = 0; i < network.hiddenLayer.length; i++){
			double[][] hiddenArray = new double[network.hiddenLayer[i].numRows()][network.hiddenLayer[i].numColumns()];
			for(int j = 0; j < network.hiddenLayer[i].numRows(); j++){
				for(int k = 0; k < network.hiddenLayer[i].numColumns(); k++){
					hiddenArray[j][k] = network.hiddenLayer[i].get(j, k);
				}
			}
			hiddenLayer[i] = new DenseMatrix(hiddenArray);
		}
		double[][] outputArray = new double[network.outputLayer.numRows()][network.outputLayer.numColumns()];
		for(int j = 0; j < network.outputLayer.numRows(); j++){
			for(int k = 0; k < network.outputLayer.numColumns(); k++){
				outputArray[j][k] = network.outputLayer.get(j, k);
			}
		}
		outputLayer = new DenseMatrix(outputArray);
		m = new Matrix[network.m.length];
		for(int i = 0; i < network.m.length; i++){
			double[][] mArray = new double[network.m[i].numRows()][network.m[i].numColumns()];
			for(int j = 0; j < network.m[i].numRows(); j++){
				for(int k = 0; k < network.m[i].numColumns(); k++){
					mArray[j][k] = network.m[i].get(j, k);
				}
			}
			m[i] = new DenseMatrix(mArray);
		}
		v = new Matrix[network.v.length];
		for(int i = 0; i < network.v.length; i++){
			double[][] vArray = new double[network.v[i].numRows()][network.v[i].numColumns()];
			for(int j = 0; j < network.v[i].numRows(); j++){
				for(int k = 0; k < network.v[i].numColumns(); k++){
					vArray[j][k] = network.v[i].get(j, k);
				}
			}
			v[i] = new DenseMatrix(vArray);
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
		hiddenLayer = new Matrix[hiddenSize.length];
		Matrix[] biases = resizeBiases(this.biases, inputs);
		for(int i = 0; i < hiddenLayer.length; i++){
			if(i==0){
				hiddenLayer[i] = inputLayer.multAdd(matrices[i], biases[i].copy());
			}else{
				hiddenLayer[i] = hiddenLayer[i-1].multAdd(matrices[i], biases[i].copy());
			}
		}
		outputLayer = hiddenLayer[hiddenLayer.length-1].multAdd(matrices[matrices.length-1], biases[biases.length-1].copy());
		return outputLayer;
	}
	
	public Matrix[] gradients(Matrix inputs, Matrix expectedOutput){
		//make gradients of biases
		Matrix output = feedForward(inputs);
		Matrix error = expectedOutput.copy().add(-1, output);
		//change weight by derivative of error
		Matrix[] delta = new Matrix[matrices.length];
		Matrix[] deltaBias = new Matrix[matrices.length];
		Matrix[] dW = new Matrix[matrices.length];
		Matrix[] dWBias = new Matrix[matrices.length];
		for(int i = matrices.length-1; i >= 0; i--){
			if(i==matrices.length-1){
				Matrix derivative = dTanh(output);
				delta[i] = elementMult(error.copy().scale(-1), derivative);
				deltaBias[i] = elementMult(error.copy().scale(-1), derivative);
			}else{
				Matrix derivative = dTanh(hiddenLayer[i]);
				delta[i] = new DenseMatrix(delta[i+1].numRows(), matrices[i+1].numRows());
				delta[i] = elementMult(delta[i+1].transBmult(matrices[i+1].copy(), delta[i]), derivative);
				deltaBias[i] = new DenseMatrix(delta[i+1].numRows(), matrices[i+1].numRows());
				deltaBias[i] = elementMult(delta[i+1].transBmult(matrices[i+1].copy(), deltaBias[i]), derivative);
			}
			if(i!=0){
				dW[i] = new DenseMatrix(matrices[i].numRows(), matrices[i].numColumns());
				dW[i] = tanh(hiddenLayer[i-1]).copy().transAmult(delta[i], dW[i]);
				dWBias[i] = new DenseMatrix(biases[i].numRows(), biases[i].numColumns());
				Matrix ones = new DenseMatrix(deltaBias[i].numRows(), biases[i].numRows());
				for(int j = 0; j < ones.numRows(); j++){
					for(int k = 0; k < ones.numColumns(); k++){
						ones.set(j, k, 1);
					}
				}
				dWBias[i] = ones.transAmult(deltaBias[i], dWBias[i]);
			}else{
				dW[i] = new DenseMatrix(matrices[i].numRows(), matrices[i].numColumns());
				dW[i] = tanh(inputLayer).copy().transAmult(delta[i], dW[i]);
				dWBias[i] = new DenseMatrix(biases[i].numRows(), biases[i].numColumns());
				Matrix ones = new DenseMatrix(deltaBias[i].numRows(), biases[i].numRows());
				for(int j = 0; j < ones.numRows(); j++){
					for(int k = 0; k < ones.numColumns(); k++){
						ones.set(j, k, 1);
					}
				}
				dWBias[i] = ones.transAmult(deltaBias[i], dWBias[i]);
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
		for(int i = 0; i < matrices.length; i++){
			matrices[i] = matrices[i].add(-1, dW[i].copy().scale(learningRate));
		}
	}
	
	public Matrix[] adam(double[][] inputs, double[][] expectedOutput, double learningRate, double b1, double b2, double e){
		t++;
		Matrix[] g = gradients(new DenseMatrix(inputs), new DenseMatrix(expectedOutput));
		int count = 0;
		for(int i = 0; i < matrices.length + biases.length; i++){
			m[i] = m[i].copy().scale(b1).add(g[i].copy().scale(1-b1));
			v[i] = v[i].copy().scale(b2).add(elementMult(g[i], g[i]).scale(1-b2));
			double alpha = learningRate * Math.sqrt(1 - Math.pow(b2, t))/(1-Math.pow(b1, t));
			if(i%2==0){
				matrices[count] = matrices[count].copy().add(-1, elementAdam(alpha, m[i], v[i], e));
			}else{
				biases[count] = biases[count].copy().add(-1, elementAdam(alpha, m[i], v[i], e));
				count++;
			}
		}
		return g;
	}
	
	public double verify(double[][] inputs, int[] expectedOutput){
		double accuracy = 0;
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
		}
		accuracy = accuracy / inputs.length;
		return accuracy;
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
		
		Matrix[] newMatrices = new Matrix[matrices.length+1];
		for(int i = 0; i < matrices.length; i++){
			newMatrices[i] = matrices[i];
		}
		newMatrices[newMatrices.length-1] = initWeights(outputSize, outputSize);
		matrices = newMatrices;
		
		Matrix[] newBiases = new Matrix[biases.length+1];
		for(int i = 0; i < biases.length; i++){
			newBiases[i] = biases[i];
		}
		newBiases[newBiases.length-1] = initWeights(1, outputSize);
		biases = newBiases;		
		
		Matrix[] newM = new Matrix[m.length+2];
		for(int i = 0; i < m.length; i++){
			newM[i] = m[i];
		}
		newM[newM.length-2] = (new DenseMatrix(outputSize, outputSize)).zero();
		newM[newM.length-1] = (new DenseMatrix(1, outputSize)).zero();
		m = newM;
		
		Matrix[] newV = new Matrix[v.length+2];
		for(int i = 0; i < v.length; i++){
			newV[i] = v[i];
		}
		newV[newV.length-2] = (new DenseMatrix(outputSize, outputSize)).zero();
		newV[newV.length-1] = (new DenseMatrix(1, outputSize)).zero();
		v = newV;
	}
	
	//layernum = the layer that gets the new neuron.  Input is 1, output is last
	public void addNeuron(int layerNum){
		hiddenSize[layerNum-2]++;
		matrices[layerNum-2] = addColumn(matrices[layerNum-2], true);
		matrices[layerNum-1] = addRow(matrices[layerNum-1], true);
		biases[layerNum-2] = addColumn(biases[layerNum-2], true);
		m[2*(layerNum-2)] = addColumn(m[2*(layerNum-2)], false);
		m[2*(layerNum-1)] = addRow(m[2*(layerNum-1)], false);
		m[2*(layerNum-2)+1] = addColumn(m[2*(layerNum-2)+1], false);
		v[2*(layerNum-2)] = addColumn(v[2*(layerNum-2)], false);
		v[2*(layerNum-1)] = addRow(v[2*(layerNum-1)], false);
		v[2*(layerNum-2)+1] = addColumn(v[2*(layerNum-2)+1], false);
	}
	
	public Matrix addRow(Matrix matrix, boolean isNetwork){
		int rows = matrix.numRows();
		int col = matrix.numColumns();
		double[][] weights = new double[rows+1][col];
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < col; j++){
				weights[i][j] = matrix.get(i, j);
			}
		}
		double[][] newRow;
		if(isNetwork){
			newRow = initValues(1, col);
		}else{
			newRow = new double[1][col];
			for(int i = 0; i < col; i++){
				newRow[0][i] = 0;
			}
		}
		for(int i = 0; i < col; i++){
			weights[weights.length-1][i] = newRow[0][i];
		}
		Matrix newMatrix = new DenseMatrix(weights);
		return newMatrix;
	}
	
	public Matrix addColumn(Matrix matrix, boolean isNetwork){
		int rows = matrix.numRows();
		int col = matrix.numColumns();
		double[][] weights = new double[rows][col+1];
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < col; j++){
				weights[i][j] = matrix.get(i, j);
			}
		}
		double[][] newCol;
		if(isNetwork){
			newCol = initValues(rows, 1);
		}else{
			newCol = new double[rows][1];
			for(int i = 0; i < rows; i++){
				newCol[i][0] = 0;
			}
		}
		for(int i = 0; i < rows; i++){
			weights[i][weights[i].length-1] = newCol[i][0];
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
	
	public Matrix[] resizeBiases(Matrix[] biases, Matrix inputs){
		double[][][] resizedArray = new double[biases.length][][];
		Matrix[] resized = new Matrix[biases.length];
		for(int n = 0; n < biases.length; n++){
			resizedArray[n] = new double[inputs.numRows()][biases[n].numColumns()];
			for(int i = 0; i < resizedArray[n].length; i++){
				for(int j = 0; j < resizedArray[n][i].length; j++){
					resizedArray[n][i][j] = biases[n].get(0, j);
				}
			}
			resized[n] = new DenseMatrix(resizedArray[n]);
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
		for(int i = 0; i < matrices.length; i++){
			for(int j = 0; j < matrices[i].numRows(); j++){
				for(int k = 0; k < matrices[i].numColumns(); k++){
					output += matrices[i].get(j, k);
					if(k!=matrices[i].numColumns() - 1){
						output += " ";
					}
				}
				output += System.lineSeparator();
			}
			System.out.println("Finished matrix " + i);
			for(int j = 0; j < biases[i].numRows(); j++){
				for(int k = 0; k < biases[i].numColumns(); k++){
					output += biases[i].get(j, k);
					if(k!=biases[i].numColumns()-1){
						output += " ";
					}
				}
				if(!(i==matrices.length-1&&j==biases[i].numRows()-1)){
					output += System.lineSeparator();
				}
			}
			System.out.println("Finished bias " + i);
		}
		return output;
	}
	
	public GrowingNN clone(){
		GrowingNN clone = new GrowingNN(inputSize, hiddenSize, outputSize);
		clone.matrices = new Matrix[matrices.length];
		clone.biases = new Matrix[biases.length];
		for(int i = 0; i < matrices.length; i++){
			clone.matrices[i] = matrices[i].copy();
			clone.biases[i] = biases[i].copy();
		}
		clone.inputLayer = inputLayer.copy();
		clone.hiddenLayer = new Matrix[hiddenLayer.length];
		for(int i = 0; i < hiddenLayer.length; i++){
			clone.hiddenLayer[i] = hiddenLayer[i].copy();
		}
		clone.outputLayer = outputLayer.copy();
		clone.m = new Matrix[m.length];
		clone.v = new Matrix[v.length];
		for(int i = 0; i < m.length; i++){
			clone.m[i] = m[i].copy();
			clone.v[i] = v[i].copy();
		}
		clone.t = t;
		clone.learningRate = learningRate;
		clone.b1 = b1;
		clone.b2 = b2;
		clone.e = e;
		return clone;
	}

}
