package com.bezold.nn.growing_nn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import org.apache.logging.log4j.Logger;
import org.jblas.FloatMatrix;

public class GrowingNN implements Cloneable{
	
	int inputSize;
	int[] hiddenSize;
	int outputSize;
	
	ArrayList<FloatMatrix> matrices;
	ArrayList<FloatMatrix> biases;
	
	FloatMatrix inputLayer;
	ArrayList<FloatMatrix> hiddenLayer;
	FloatMatrix outputLayer;
	
	ArrayList<FloatMatrix> m;
	ArrayList<FloatMatrix> v;
	int t;
	
	float learningRate = .001f;
	float b1 = .9f;
	float b2 = .999f;
	float e = .0001f;
	float l1parameter = .001f;
	
	float dropoutRate = .2f;
	Random random;
	
	static final int TANH = 0;
	static final int SIGMOID = 1;
	static final int RELU = 2;
	
	int logloss = 0;
	
	int[] layerActivation;
	
	public GrowingNN(int inputSize, int hiddenSize, int outputSize){
		this(inputSize, new int[]{hiddenSize}, outputSize);
	}
	
	public GrowingNN(int inputSize, int[] hiddenSize, int outputSize){
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.outputSize = outputSize;
		
		matrices = new ArrayList<FloatMatrix>(hiddenSize.length+1);
		for(int i = 0; i < hiddenSize.length+1; i++){
			if(i==0){
				matrices.add(initWeights(inputSize, hiddenSize[0]));
			}else if(i==hiddenSize.length){
				matrices.add(initWeights(hiddenSize[i-1], outputSize));
			}else{
				matrices.add(initWeights(hiddenSize[i-1], hiddenSize[i]));
			}
		}
		
		biases = new ArrayList<FloatMatrix>(hiddenSize.length+1);
		for(int i = 0; i < hiddenSize.length+1; i++){
			if(i==hiddenSize.length){
				biases.add(initWeights(1, outputSize));
			}else{
				biases.add(initWeights(1, hiddenSize[i]));
			}
		}
		
		m = new ArrayList<FloatMatrix>(matrices.size() + biases.size());
		v = new ArrayList<FloatMatrix>(matrices.size() + biases.size());
		int count = 0;
		for(int i = 0; i < matrices.size()+biases.size(); i++){
			if(i%2==0){
				m.add(FloatMatrix.zeros(matrices.get(count).rows, matrices.get(count).columns));
				v.add(FloatMatrix.zeros(matrices.get(count).rows, matrices.get(count).columns));
			}else{
				m.add(FloatMatrix.zeros(biases.get(count).rows, biases.get(count).columns));
				v.add(FloatMatrix.zeros(biases.get(count).rows, biases.get(count).columns));
				count++;
			}
		}
		t = 0;
		random = new Random();
		
		layerActivation = new int[hiddenSize.length+1];
		for(int i = 0; i < layerActivation.length; i++){
			layerActivation[i] = 0;
		}
	}
	
	public GrowingNN(GrowingNN network){
		inputSize = network.inputSize;
		hiddenSize = new int[network.hiddenSize.length];
		for(int i = 0; i < network.hiddenSize.length; i++){
			hiddenSize[i] = network.hiddenSize[i];
		}
		outputSize = network.outputSize;
		
		matrices = new ArrayList<FloatMatrix>(network.matrices.size());
		for(int i = 0; i < network.matrices.size(); i++){
			matrices.add(network.matrices.get(i).dup());
		}
		biases = new ArrayList<FloatMatrix>(network.biases.size());
		for(int i = 0; i < network.biases.size(); i++){
			biases.add(network.biases.get(i).dup());
		}
		
		inputLayer = network.inputLayer.dup();
		hiddenLayer = new ArrayList<FloatMatrix>(network.hiddenLayer.size());
		for(int i = 0; i < network.hiddenLayer.size(); i++){
			hiddenLayer.add(network.hiddenLayer.get(i).dup());
		}
		outputLayer = network.outputLayer.dup();
		
		m = new ArrayList<FloatMatrix>(network.m.size());
		for(int i = 0; i < network.m.size(); i++){
			m.add(network.m.get(i).dup());
		}
		v = new ArrayList<FloatMatrix>(network.v.size());
		for(int i = 0; i < network.v.size(); i++){
			v.add(network.v.get(i).dup());
		}
		
		t = network.t;
		learningRate = network.learningRate;
		b1 = network.b1;
		b2 = network.b2;
		e = network.e;
		random = new Random();
		
	}
	
	public float[][] feedForward(float[][] inputs, float dropoutRate, boolean dropout){
		inputLayer = new FloatMatrix(inputs);
		outputLayer = feedForward(inputLayer, dropoutRate, dropout);
		float[][] outputs = outputLayer.toArray2();
		return outputs;
	}
	
	public FloatMatrix feedForward(FloatMatrix inputs, float dropoutRate, boolean dropout){
		inputLayer = inputs;
		hiddenLayer = new ArrayList<FloatMatrix>(hiddenSize.length);
		ArrayList<FloatMatrix> biases = resizeBiases(this.biases, inputs);
		for(int i = 0; i < hiddenSize.length; i++){
			if(i==0){
				hiddenLayer.add(inputLayer.mmul(matrices.get(i)).add(biases.get(i)));
			}else{
				hiddenLayer.add(activate(hiddenLayer.get(i-1), i-1).mmul(matrices.get(i)).add(biases.get(i)));
			}
			if(dropout){
				for(int j = 0; j < hiddenLayer.get(i).rows; j++){
					for(int k = 0; k < hiddenLayer.get(i).columns; k++){
						if(random.nextDouble() < dropoutRate){
							hiddenLayer.get(i).put(j, k, 0);
						}
					}
				}
				hiddenLayer.get(i).muli(1/(1-dropoutRate));
			}
		}
		outputLayer = activate(hiddenLayer.get(hiddenLayer.size()-1), hiddenLayer.size()-1).mmul(matrices.get(matrices.size()-1)).add(biases.get(biases.size()-1));
		return activate(outputLayer, layerActivation.length-1);
	}
	
	public FloatMatrix[] gradients(FloatMatrix inputs, FloatMatrix expectedOutput){
		//make gradients of biases
		FloatMatrix output = feedForward(inputs, dropoutRate, true);
		if(Float.isNaN(output.get(0, 0))){
			System.out.println("NaN in output");
			System.exit(1);
		}
		//error = output - expectedOutput
		FloatMatrix error;
		if(logloss == 0){
			error = output.sub(expectedOutput);
		}else{
			FloatMatrix logOutput = new FloatMatrix(output.rows, output.columns);
			FloatMatrix negLogOutput = new FloatMatrix(output.rows, output.columns);;
			for(int i = 0; i < logOutput.rows; i++){
				for(int j = 0; j < logOutput.columns; j++){
					if(output.get(i, j) >= 1){
						output.put(i, j, .999999f);
					}
					if(output.get(i, j) <= 0){
						output.put(i, j, .000001f);
					}
					float log = (float) Math.log(output.get(i, j));
					float negLog = (float) Math.log(1-output.get(i, j));
					logOutput.put(i, j, log);
					negLogOutput.put(i, j, negLog);
				}
			}
			FloatMatrix ones2 = FloatMatrix.ones(expectedOutput.rows, expectedOutput.columns);
			error = expectedOutput.mul(logOutput).sub(ones2.sub(expectedOutput).mul(negLogOutput));
		}
		boolean flag = false;
		//apply L1 Regularization
		/*
		double l1 = 0;
		int l1count = 0;
		for(int i = 0; i < matrices.size(); i++){
			for(int j = 0; j < matrices.get(i).rows; j++){
				for(int k = 0; k < matrices.get(i).columns; k++){
					l1 += Math.abs(matrices.get(i).get(j, k));
					l1count++;
				}
			}
		}
		l1 *= l1parameter/l1count;
		double[][] l1MatrixSet = new double[error.rows][error.columns];
		for(int i = 0; i < l1MatrixSet.length; i++){
			for(int j = 0; j < l1MatrixSet[i].length; j++){
				l1MatrixSet[i][j] = l1;
			}
		}
		Matrix l1Matrix = new DenseMatrix(l1MatrixSet);
		error.add(l1Matrix);
		*/
		//change weight by derivative of error
		FloatMatrix[] delta = new FloatMatrix[matrices.size()];
		FloatMatrix[] deltaBias = new FloatMatrix[matrices.size()];
		FloatMatrix[] dW = new FloatMatrix[matrices.size()];
		FloatMatrix[] dWBias = new FloatMatrix[matrices.size()];
		for(int i = matrices.size()-1; i >= 0; i--){
			if(i==matrices.size()-1){
				FloatMatrix derivative = derivative(outputLayer, layerActivation.length-1);
				delta[i] = error.mul(derivative);
				deltaBias[i] = delta[i].dup();
			}else{
				FloatMatrix derivative = derivative(hiddenLayer.get(i), i);
				delta[i] = delta[i+1].mmul(matrices.get(i+1).transpose()).mul(derivative);
				deltaBias[i] = delta[i+1].mmul(matrices.get(i+1).transpose()).mul(derivative);
			}
			if(i!=0){
				dW[i] = activate(hiddenLayer.get(i-1), i-1).transpose().mmul(delta[i]);
				FloatMatrix ones = FloatMatrix.ones(deltaBias[i].rows, biases.get(i).rows);
				dWBias[i] = ones.transpose().mmul(deltaBias[i]);
			}else{
				dW[i] = inputLayer.transpose().mmul(delta[i]);
				FloatMatrix ones = FloatMatrix.ones(deltaBias[i].rows, biases.get(i).rows);
				dWBias[i] = ones.transpose().mmul(deltaBias[i]);
			}
		}
		
		FloatMatrix avgError = FloatMatrix.ones(1, error.rows).mul((1/error.rows)).mmul(error);
		FloatMatrix[] returned = new FloatMatrix[dW.length + dWBias.length + 1];
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
	
	public void backpropagate(float[][] inputs, float[][] expectedOutput, float learningRate){
		FloatMatrix[] dW = gradients(new FloatMatrix(inputs), new FloatMatrix(expectedOutput));
		for(int i = 0; i < matrices.size(); i++){
			matrices.get(i).subi(dW[i].mul(learningRate));
		}
	}
	
	public FloatMatrix[] adam(float[][] inputs, float[][] expectedOutput, float learningRate, float b1, float b2, float e){
		t++;
		FloatMatrix[] g = gradients(new FloatMatrix(inputs), new FloatMatrix(expectedOutput));
		int count = 0;
		for(int i = 0; i < matrices.size() + biases.size(); i++){
			m.get(i).muli(b1).addi(g[i].mul(1-b1));
			v.get(i).muli(b2).addi(g[i].mul(g[i]).mul(1-b2));
			double alpha = learningRate * Math.sqrt(1 - Math.pow(b2, t))/(1-Math.pow(b1, t));
			if(i%2==0){
				matrices.get(count).subi(elementAdam(alpha, m.get(i), v.get(i), e));
			}else{
				biases.get(count).subi(elementAdam(alpha, m.get(i), v.get(i), e));
				count++;
			}
		}
		return g;
	}
	
	public void train(float[][] inputs, float[][] expectedOutput, float[][] verifyInputs, float[][] verifyExpectedOutput, int epochs, int batchSize, Logger log){
		int iterator = 0;
		Integer[] shuffled = new Integer[inputs.length];
		for(int i = 0; i < shuffled.length; i++){
			shuffled[i] = Integer.valueOf(i);
		}
		shuffle(shuffled);
		int epoch = 0;
		boolean endOfEpoch = false;
		int minDataBeforeGrow = 100;
		Integer[] batch;
		int lastGrowth = 0;
		for(int num = 0; num < inputs.length * epochs + batchSize; num += batchSize){
			if((num/batchSize)%1000==0){
				log.info("Batch: " + (num/batchSize));
			}
			int thisSize;
			if(num + batchSize > inputs.length * epochs){
				thisSize = inputs.length * epochs - num;
			}else{
				thisSize = batchSize;
			}
			if(thisSize > 0){
				batch = new Integer[thisSize];
				for(int i = 0; i < thisSize; i++){
					if(iterator >= inputs.length){
						shuffle(shuffled);
						iterator = 0;
						epoch++;
						endOfEpoch = true;
					}
					batch[i] = shuffled[iterator];
					iterator++;
				}
				if(endOfEpoch){
					endOfEpoch = false;
					if(num - lastGrowth >= minDataBeforeGrow){
						log.info("Epoch: " + epoch);
						lastGrowth = num;
					}
				}
				float[][] batchInputs = new float[batch.length][inputSize];
				float[][] batchOutputs = new float[batch.length][outputSize];
				for(int i = 0; i < batch.length; i++){
					batchInputs[i] = inputs[batch[i]];
					batchOutputs[i] = expectedOutput[batch[i]];
				}
				adam(batchInputs, batchOutputs, learningRate, b1, b2, e);
			}
		}
		
	}
	
	public void grow(float[][] inputs, float[][] expectedOutput, float[][] verifyInput, float[][] verifyOutput, int epochs, int batchSize, Logger log){
		int iterator = 0;
		Integer[] shuffled = new Integer[inputs.length];
		for(int i = 0; i < shuffled.length; i++){
			shuffled[i] = Integer.valueOf(i);
		}
		shuffle(shuffled);
		int epoch = 0;
		boolean endOfEpoch = false;
		int minDataBeforeGrow = 100;
		Integer[] batch;
		int lastGrowth = 0;
		int newLayerNeurons = 0;
		double startError = 0;
		double currentError = 0;
		for(int num = 0; num < inputs.length * epochs + batchSize; num += batchSize){
			int thisSize;
			if(num + batchSize > inputs.length * epochs){
				thisSize = inputs.length * epochs - num;
			}else{
				thisSize = batchSize;
			}
			if(thisSize > 0){
				batch = new Integer[thisSize];
				for(int i = 0; i < thisSize; i++){
					if(iterator >= inputs.length){
						shuffle(shuffled);
						iterator = 0;
						epoch++;
						endOfEpoch = true;
					}
					batch[i] = shuffled[iterator];
					iterator++;
				}
				if(endOfEpoch){
					endOfEpoch = false;
					if(num - lastGrowth >= minDataBeforeGrow){
						//prune network
						/*
						for(int i = 0; i < matrices.size() - 1; i++){
							for(int j = 0; j < matrices.get(i).columns; j++){
								boolean deleteNeuron1 = true;
								boolean deleteNeuron2 = true;
								for(int k = 0; k < matrices.get(i).rows; k++){
									if(matrices.get(i).get(k, j) != 0){
										deleteNeuron1 = false;
									}
								}
								for(int k = 0; k < matrices.get(i+1).columns; k++){
									if(matrices.get(i+1).get(j, k) != 0){
										deleteNeuron2 = false;
									}
								}
								if(deleteNeuron1 && deleteNeuron2){
									//delete from matices.get(i)
									//delete from matrices.get(i+1)
									//delete from biases
									//delete from m
									//delete from v
									hiddenSize[i] -= 1;
									matrices.set(i, deleteColumn(matrices.get(i), j));
									matrices.set(i+1, deleteRow(matrices.get(i+1), j));
									biases.set(i, deleteColumn(biases.get(i), j));
									m.set(2*(i), deleteColumn(m.get(2*(i)), j));
									m.set(2*(i+1), deleteRow(m.get(2*(i+1)), j));
									m.set(2*(i)+1, deleteColumn(m.get(2*(i)+1), j));
									v.set(2*(i), deleteColumn(v.get(2*(i)), j));
									v.set(2*(i+1), deleteRow(v.get(2*(i+1)), j));
									v.set(2*(i)+1, deleteColumn(v.get(2*(i)+1), j));
									j--;
								}
							}
							//if a layer is emptied
							if(matrices.get(i).columns == 0){
								int[] newHiddenSize = new int[hiddenSize.length-1];
								for(int j = 0; j < newHiddenSize.length; j++){
									int newj = j;
									if(j>=i){
										newj++;
									}
									newHiddenSize[j] = hiddenSize[newj];
								}
								hiddenSize = newHiddenSize;
								
								matrices.remove(i+1);
								int rows;
								int columns;
								if(i == 0){
									rows = inputSize;
								}else{
									rows = hiddenSize[i-1];
								}
								if(i == matrices.size()-1){
									columns = outputSize;
								}else{
									columns = hiddenSize[i];
								}
								matrices.set(i, initWeights(rows, columns));
								
								biases.remove(i);
								
								m.remove(2*(i+1));
								m.remove(2*(i)+1);
								m.set(2*i, (new DenseMatrix(rows, columns)).zero());
								
								v.remove(2*(i+1));
								v.remove(2*(i)+1);
								v.set(2*i, (new DenseMatrix(rows, columns)).zero());
							}
						}
						*/
						//grow network
						int numNeurons = (int) (500*(1/(1+Math.exp((5/outputSize)*((currentError-startError)-(outputSize/2)))))*(1/(1+Math.exp(-(5/outputSize)*((currentError)+(outputSize/2))))));
						boolean keepAdding = true;
						int layer = 2;
						while(keepAdding){
							if(layer < hiddenSize.length+2){
								addNeuron(layer, numNeurons);
								layer++;
							}else{
								newLayerNeurons += numNeurons;
								if(newLayerNeurons >= outputSize){
									addLayer();
									newLayerNeurons = 0;
								}
								keepAdding = false;
							}
							numNeurons /= 5;
						}
						log.info("Epoch: " + epoch);
						float output0 = feedForward(new float[][]{{0,0}}, 0, false)[0][0];
						float output1 = feedForward(new float[][]{{0,1}}, 0, false)[0][0];
						float output2 = feedForward(new float[][]{{1,0}}, 0, false)[0][0];
						float output3 = feedForward(new float[][]{{1,1}}, 0, false)[0][0];
						String networkSize = "";
						networkSize += inputSize + " ";
						for(int j = 0; j < hiddenSize.length; j++){
							networkSize += hiddenSize[j] + " ";
						}
						networkSize += outputSize;
						float error = verify(inputs, expectedOutput)[0];
						log.info("Size: " + networkSize);
						log.info("Error: " + error);
						System.out.println("0 0: " + output0);
						System.out.println("0 1: " + output1);
						System.out.println("1 0: " + output2);
						System.out.println("1 1: " + output3);
						startError = 0;
						lastGrowth = num;
					}
				}
				float[][] batchInputs = new float[batch.length][inputSize];
				float[][] batchOutputs = new float[batch.length][outputSize];
				for(int i = 0; i < batch.length; i++){
					batchInputs[i] = inputs[batch[i]];
					batchOutputs[i] = expectedOutput[batch[i]];
				}
				adam(batchInputs, batchOutputs, learningRate, b1, b2, e);
				currentError = verify(verifyInput, verifyOutput)[0];
				if(startError == 0){
					startError = currentError;
				}
			}
		}
		
	}
	
	public float[] verify(float[][] inputs, int[] expectedOutput){
		float accuracy = 0;
		float absError = 0;
		float[][] output = feedForward(inputs, 0, false);
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
		return new float[]{accuracy, absError};
	}
	
	public float[] verify(float[][] inputs, float[][] expectedOutput){
		float absError = 0;
		float[][] output = feedForward(inputs, 0, false);
		for(int i = 0; i < output.length; i++){
			for(int j = 0; j < output[i].length; j++){
				absError += Math.abs(expectedOutput[i][j] - output[i][j]);
			}
		}
		float totalError = absError;
		absError /= inputs.length;
		return new float[]{absError, totalError};
		/*FloatMatrix output = new FloatMatrix(outputArray);
		FloatMatrix logOutput = new FloatMatrix(output.rows, output.columns);
		FloatMatrix negLogOutput = new FloatMatrix(output.rows, output.columns);;
		for(int i = 0; i < logOutput.rows; i++){
			for(int j = 0; j < logOutput.columns; j++){
				if(output.get(i, j) >= 1){
					output.put(i, j, .999999f);
				}
				if(output.get(i, j) <= 0){
					output.put(i, j, .000001f);
				}
				float log = (float) Math.log(output.get(i, j));
				float negLog = (float) Math.log(1-output.get(i, j));
				logOutput.put(i, j, log);
				negLogOutput.put(i, j, negLog);
			}
		}
		FloatMatrix expectedOutputMatrix = new FloatMatrix(expectedOutput);
		FloatMatrix ones2 = FloatMatrix.ones(expectedOutputMatrix.rows, expectedOutputMatrix.columns);
		FloatMatrix error = (expectedOutputMatrix.mul(logOutput).add(ones2.sub(expectedOutputMatrix).mul(negLogOutput))).neg();
		for(int i = 0; i < error.rows; i++){
			for(int j = 0; j < error.columns; j++){
				absError += error.get(i, j);
			}
		}
		float totalError = absError;
		absError /= inputs.length;
		return new float[]{absError, totalError};*/
	}
	
	//set weights to between -0.1 and 0.1
	public FloatMatrix initWeights(int rows, int columns){
		FloatMatrix matrix = FloatMatrix.rand(rows, columns).mul(.2f).sub(.1f);
		return matrix;
	}
	
	public void addLayer(){
		int newLayerSize = outputSize;
		if(outputSize < 3){
			newLayerSize = 3;
		}
		int[] newHiddenSize = new int[hiddenSize.length+1];
		for(int i = 0; i < hiddenSize.length; i++){
			newHiddenSize[i] = hiddenSize[i];
		}
		newHiddenSize[newHiddenSize.length-1] = newLayerSize;
		hiddenSize = newHiddenSize;
		
		matrices.set(matrices.size()-1, addColumn(matrices.get(matrices.size()-1), newLayerSize-outputSize, true));
		matrices.add(initWeights(newLayerSize, outputSize));
		
		biases.add(biases.size()-1, initWeights(1, newLayerSize));
		
		m.set(m.size()-2, addColumn(m.get(m.size()-2), newLayerSize-outputSize, false));
		m.add(m.size()-1, (FloatMatrix.zeros(newLayerSize, outputSize)));
		m.add(m.size()-2, (FloatMatrix.zeros(1, newLayerSize)));
		
		v.set(v.size()-2, addColumn(v.get(v.size()-2), newLayerSize-outputSize, false));
		v.add(v.size()-1, (FloatMatrix.zeros(newLayerSize, outputSize)));
		v.add(v.size()-2, (FloatMatrix.zeros(1, newLayerSize)));
	}
	
	//layernum = the layer that gets the new neuron.  Input is 1, output is last
	public void addNeuron(int layerNum, int numNeurons){
		hiddenSize[layerNum-2] += numNeurons;
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
	
	public FloatMatrix addRow(FloatMatrix matrix, int numNeurons, boolean isNetwork){
		int rows = matrix.rows;
		int col = matrix.columns;
		float[][] weights = new float[rows+numNeurons][col];
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < col; j++){
				weights[i][j] = matrix.get(i, j);
			}
		}
		float[][] newRow;
		if(isNetwork){
			newRow = initValues(numNeurons, col);
		}else{
			newRow = new float[numNeurons][col];
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
		FloatMatrix newMatrix = new FloatMatrix(weights);
		return newMatrix;
	}
	
	public FloatMatrix addColumn(FloatMatrix matrix, int numNeurons, boolean isNetwork){
		int rows = matrix.rows;
		int col = matrix.columns;
		float[][] weights = new float[rows][col+numNeurons];
		for(int i = 0; i < rows; i++){
			for(int j = 0; j < col; j++){
				weights[i][j] = matrix.get(i, j);
			}
		}
		float[][] newCol;
		if(isNetwork){
			newCol = initValues(rows, numNeurons);
		}else{
			newCol = new float[rows][numNeurons];
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
		FloatMatrix newMatrix = new FloatMatrix(weights);
		return newMatrix;
	}
	
	public FloatMatrix deleteRow(FloatMatrix matrix, int row){
		float[][] newMatrixWeights = new float[matrix.rows-1][matrix.columns];
		for(int i = 0; i < newMatrixWeights.length; i++){
			for(int j = 0; j < newMatrixWeights[i].length; j++){
				int newi = i;
				if(i >= row){
					newi++;
				}
				newMatrixWeights[i][j] = matrix.get(newi, j);
			}
		}
		FloatMatrix newMatrix = new FloatMatrix(newMatrixWeights);
		return newMatrix;
	}
	
	public FloatMatrix deleteColumn(FloatMatrix matrix, int column){
		float[][] newMatrixWeights = new float[matrix.rows][matrix.columns-1];
		for(int i = 0; i < newMatrixWeights.length; i++){
			for(int j = 0; j < newMatrixWeights[i].length; j++){
				int newj = j;
				if(j >= column){
					newj++;
				}
				newMatrixWeights[i][j] = matrix.get(i, newj);
			}
		}
		FloatMatrix newMatrix = new FloatMatrix(newMatrixWeights);
		return newMatrix;
	}
	
	public float[][] initValues(int rows, int col){
		float[][] weights = new float[col][rows];
		for(int i = 0; i < weights.length; i++){
			for(int j = 0; j < weights[i].length; j++){
				weights[i][j] = (float) (Math.random() * .2 - .1);
			}
		}
		return weights;
	}
	
	public FloatMatrix activate(FloatMatrix x, int layer){
		switch(layerActivation[layer]){
		case 0:
			return tanh(x);
		case 1:
			return sigmoid(x);
		case 2:
			return relu(x);
		default:
			//turn this into an exception and log it in the train function
			System.out.println("Bad Activation Function, layer " + (layer+1));
			System.exit(1);
			return null;
		}
	}
	
	public FloatMatrix derivative(FloatMatrix x, int layer){
		switch(layerActivation[layer]){
		case 0:
			return dTanh(x);
		case 1:
			return dSigmoid(x);
		case 2:
			return dRelu(x);
		default:
			System.out.println("Bad Activation Function, layer " + (layer+1));
			System.exit(1);
			return null;
		}
	}
	
	public FloatMatrix tanh(FloatMatrix x){
		float[][] value = new float[x.rows][x.columns];
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				value[i][j] = (float) Math.tanh(x.get(i, j));
			}
		}
		FloatMatrix returnMatrix = new FloatMatrix(value);
		return returnMatrix;
	}
	
	public FloatMatrix dTanh(FloatMatrix x){
		float[][] value = new float[x.rows][x.columns];
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				value[i][j] = (float) (1 - Math.pow(Math.tanh(x.get(i, j)), 2));
			}
		}
		FloatMatrix returnMatrix = new FloatMatrix(value);
		return returnMatrix;
	}
	
	public FloatMatrix sigmoid(FloatMatrix x){
		float[][] value = new float[x.rows][x.columns];
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				value[i][j] = (float) (1/(1+Math.exp(-1*x.get(i, j))));
			}
		}
		FloatMatrix returnMatrix = new FloatMatrix(value);
		return returnMatrix;
	}
	
	public FloatMatrix dSigmoid(FloatMatrix x){
		float[][] value = new float[x.rows][x.columns];
		FloatMatrix sigmoid = sigmoid(x);
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				value[i][j] = (float) (sigmoid.get(i, j) * (1-sigmoid.get(i, j)));
			}
		}
		FloatMatrix returnMatrix = new FloatMatrix(value);
		return returnMatrix;
	}
	
	public FloatMatrix relu(FloatMatrix x){
		float[][] value = new float[x.rows][x.columns];
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				value[i][j] = Math.max(0, x.get(i, j));
			}
		}
		FloatMatrix returnMatrix = new FloatMatrix(value);
		return returnMatrix;
	}
	
	public FloatMatrix dRelu(FloatMatrix x){
		float[][] value = new float[x.rows][x.columns];
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				if(x.get(i, j) >= 0){
					value[i][j] = 1;
				}else{
					value[i][j] = 0;
				}
			}
		}
		FloatMatrix returnMatrix = new FloatMatrix(value);
		return returnMatrix;
	}
	
	
	public FloatMatrix elementAdam(double alpha, FloatMatrix m, FloatMatrix v, double e){
		float[][] value = new float[m.rows][m.columns];
		for(int i = 0; i < value.length; i++){
			for(int j = 0; j < value[i].length; j++){
				value[i][j] = (float) (alpha * m.get(i, j) / (Math.sqrt(v.get(i, j)) + e));
			}
		}
		FloatMatrix resultMatrix = new FloatMatrix(value);
		return resultMatrix;
	}
	
	public ArrayList<FloatMatrix> resizeBiases(ArrayList<FloatMatrix> biases, FloatMatrix inputs){
		float[][][] resizedArray = new float[biases.size()][][];
		ArrayList<FloatMatrix> resized = new ArrayList<FloatMatrix>(biases.size());
		for(int n = 0; n < biases.size(); n++){
			resizedArray[n] = new float[inputs.rows][biases.get(n).columns];
			for(int i = 0; i < resizedArray[n].length; i++){
				for(int j = 0; j < resizedArray[n][i].length; j++){
					resizedArray[n][i][j] = biases.get(n).get(0, j);
				}
			}
			resized.add(new FloatMatrix(resizedArray[n]));
		}
		return resized;
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
			for(int j = 0; j < matrices.get(i).rows; j++){
				for(int k = 0; k < matrices.get(i).columns; k++){
					output += matrices.get(i).get(j, k);
					if(k!=matrices.get(i).columns - 1){
						output += " ";
					}
				}
				output += System.lineSeparator();
			}
			System.out.println("Finished matrix " + i);
			for(int j = 0; j < biases.get(i).rows; j++){
				for(int k = 0; k < biases.get(i).columns; k++){
					output += biases.get(i).get(j, k);
					if(k!=biases.get(i).columns-1){
						output += " ";
					}
				}
				if(!(i==matrices.size()-1&&j==biases.get(i).rows-1)){
					output += System.lineSeparator();
				}
			}
			System.out.println("Finished bias " + i);
		}
		return output;
	}
	
	public GrowingNN clone(){
		GrowingNN clone = new GrowingNN(inputSize, hiddenSize, outputSize);
		clone.matrices = new ArrayList<FloatMatrix>(matrices.size());
		clone.biases = new ArrayList<FloatMatrix>(biases.size());
		for(int i = 0; i < matrices.size(); i++){
			clone.matrices.set(i, matrices.get(i).dup());
			clone.biases.set(i, biases.get(i).dup());
		}
		clone.inputLayer = inputLayer.dup();
		clone.hiddenLayer = new ArrayList<FloatMatrix>(hiddenLayer.size());
		for(int i = 0; i < hiddenLayer.size(); i++){
			clone.hiddenLayer.set(i, hiddenLayer.get(i).dup());
		}
		clone.outputLayer = outputLayer.dup();
		clone.m = new ArrayList<FloatMatrix>(m.size());
		clone.v = new ArrayList<FloatMatrix>(v.size());
		for(int i = 0; i < m.size(); i++){
			clone.m.set(i, m.get(i).dup());
			clone.v.set(i, v.get(i).dup());
		}
		clone.t = t;
		clone.learningRate = learningRate;
		clone.b1 = b1;
		clone.b2 = b2;
		clone.e = e;
		clone.random = new Random();
		return clone;
	}
	
	public static GrowingNN input(String filename){
		try{
			GrowingNN network;
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String encoded = reader.readLine();
			int networkSize = Integer.parseInt(encoded);
			encoded = reader.readLine();
			int inputSize = Integer.parseInt(encoded);
			int[] hiddenSize = new int[networkSize-2];
			for(int i = 0; i < hiddenSize.length; i++){
				encoded = reader.readLine();
				hiddenSize[i] = Integer.parseInt(encoded);
			}
			encoded = reader.readLine();
			int outputSize = Integer.parseInt(encoded);
			float[][][] matrixWeights = new float[networkSize-1][][];
			float[][][] biasWeights = new float[networkSize-1][][];
			for(int i = 0; i < networkSize-1; i++){
				if(i == 0){
					matrixWeights[i] = new float[inputSize][hiddenSize[0]];
					biasWeights[i] = new float[1][hiddenSize[0]];
				}else if(i == networkSize-2){
					matrixWeights[i] = new float[hiddenSize[hiddenSize.length-1]][outputSize];
					biasWeights[i] = new float[1][outputSize];
				}else{
					matrixWeights[i] = new float[hiddenSize[i-1]][hiddenSize[i]];
					biasWeights[i] = new float[1][hiddenSize[i]];
				}
				for(int j = 0; j < matrixWeights[i].length; j++){
					encoded = reader.readLine();
					String[] encodedArray = encoded.split(" ");
					for(int k = 0; k < matrixWeights[i][j].length; k++){
						matrixWeights[i][j][k] = Float.parseFloat(encodedArray[k]);
					}
				}
				for(int j = 0; j < biasWeights[i].length; j++){
					encoded = reader.readLine();
					String[] encodedArray = encoded.split(" ");
					for(int k = 0; k < biasWeights[i][j].length; k++){
						biasWeights[i][j][k] = Float.parseFloat(encodedArray[k]);
					}
				}
			}
			reader.close();
			network = new GrowingNN(inputSize, hiddenSize, outputSize);
			network.setWeights(matrixWeights, biasWeights);
			return network;
		}catch(Exception e){
			e.printStackTrace();
			return null;
		}
	}
	
	public void setWeights(float[][][] matrixWeights, float[][][] biasWeights){
		matrices.clear();
		biases.clear();
		inputSize = matrixWeights[0].length;
		hiddenSize = new int[matrixWeights.length-1];
		for(int i = 0; i < matrixWeights.length; i++){
			matrices.add(new FloatMatrix(matrixWeights[i]));
			if(i != 0){
				hiddenSize[i-1] = matrixWeights[i].length;
			}
		}
		outputSize = matrixWeights[matrixWeights.length-1][0].length;
		for(int i = 0; i < biasWeights.length; i++){
			biases.add(new FloatMatrix(biasWeights[i]));
		}
	}

}
