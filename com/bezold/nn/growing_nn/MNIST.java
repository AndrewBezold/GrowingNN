package com.bezold.nn.growing_nn;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Vector;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.ThreadContext;

import com.bezold.mnist.DigitImage;
import com.bezold.mnist.DigitImageLoadingService;

/**
 * Hello world!
 *
 */
public class MNIST 
{
	public GrowingNN[] network;
	private static final Logger log = LogManager.getLogger();
	private static final int LOW_NUMBER_OF_NETWORKS = 10;
	
    public static void main( String[] args )
    {
    	String time = LocalDateTime.now().format(DateTimeFormatter.ofPattern("uuuu-MM-dd'T'HH_mm_ss_SSS"));
    	try {
	    	ThreadContext.put("logFileName", time);
	    	String mnistTrainImageFilename = "C:/Users/Beez/Downloads/MNIST/train-images.idx3-ubyte";
	    	String mnistTrainLabelFilename = "C:/Users/Beez/Downloads/MNIST/train-labels.idx1-ubyte";
	    	String mnistTestImageFilename = "C:/Users/Beez/Downloads/MNIST/t10k-images.idx3-ubyte";
	    	String mnistTestLabelFilename = "C:/Users/Beez/Downloads/MNIST/t10k-labels.idx1-ubyte";
	    	MNIST mnist = new MNIST();
	    	mnist.network = new GrowingNN[1];
	    	mnist.network[0] = new GrowingNN(784, 3, 10);
	        	mnist.train(mnistTrainLabelFilename, mnistTrainImageFilename);
	        	mnist.test(mnistTestLabelFilename, mnistTestImageFilename);
	        	//output network
	        	mnist.network[0].output("GrowingNN" + time + ".network");
        } catch (IOException e) {
			e.printStackTrace();
		}
        
    }
    
    public void train(String labelFilename, String imageFilename) throws IOException{
        DigitImageLoadingService mnistTrainImport = new DigitImageLoadingService(labelFilename, imageFilename);
		DigitImage[] mnistFull = mnistTrainImport.loadDigitImages().toArray(new DigitImage[0]);
		DigitImage[] mnistTrain = new DigitImage[59000];
		DigitImage[] mnistVerify = new DigitImage[1000];
		for(int i = 0; i < mnistFull.length; i++){
			if(i < mnistTrain.length){
				mnistTrain[i] = mnistFull[i];
			}else{
				mnistVerify[i-mnistTrain.length] = mnistFull[i];
			}
		}
		double[][] verifyImage = new double[mnistVerify.length][784];
		double[][] verifyLabel = new double[mnistVerify.length][10];
		int[] verifyLabelNum = new int[mnistVerify.length];
		for(int i = 0; i < mnistVerify.length; i++){
			verifyImage[i] = mnistVerify[i].getData();
			verifyLabelNum[i] = mnistVerify[i].getLabel();
			for(int j = 0; j < 10; j++){
				if(j==verifyLabelNum[i]){
					verifyLabel[i][j] = 1;
				}else{
					verifyLabel[i][j] = 0;
				}
			}
		}
		shuffle(mnistTrain);
		int batchSize = 50;
		int numEpochs = 1;
		int iterator = 0;
		int epoch = 1;
		double[] layerCheck = new double[]{0};
		double[][] neuronCheck = new double[1][network[0].hiddenSize.length];
		for(int i = 0; i < neuronCheck.length; i++){
			neuronCheck[0][i] = 0;
		}
		boolean reset = false;
		boolean fullReset = false;
		DigitImage[] batch;
		for(int num = 0; num < mnistTrain.length * numEpochs + batchSize; num += batchSize){
			int thisSize;
			if(num + batchSize > mnistTrain.length * numEpochs){
				thisSize = mnistTrain.length * numEpochs - num;
			}else{
				thisSize = batchSize;
			}
			if(thisSize > 0){
				System.out.println("Batch " + ((num/batchSize) + 1) + ", Networks: " + network.length);
				batch = new DigitImage[thisSize];
				for(int i = 0; i < thisSize; i++){
					if(iterator >= mnistTrain.length){
						shuffle(mnistTrain);
						iterator = 0;
						epoch++;
					}
					batch[i] = mnistTrain[iterator];
					iterator++;
				}
				double[][] image = new double[batch.length][784];
				double[][] label = new double[batch.length][10];
				int[] labelNum = new int[batch.length];
				for(int i = 0; i < batch.length; i++){
					image[i] = batch[i].getData();
					labelNum[i] = batch[i].getLabel();
					for(int j = 0; j < 10; j++){
						if(j==labelNum[i]){
							label[i][j] = 1;
						}else{
							label[i][j] = 0;
						}
					}
				}
				//train batch
				Vector<GrowingNN> newNetworks = new Vector<GrowingNN>();
				for(int i = 0; i < network.length; i++){
					Matrix[] g = network[i].adam(image, label, network[i].learningRate, network[i].b1, network[i].b2, network[i].e);
					//verify against trained images
					double accuracy = network[i].verify(verifyImage, verifyLabelNum);
					g = network[i].gradients(new DenseMatrix(verifyImage), new DenseMatrix(verifyLabel));
					//log accuracy
					if((((num/batchSize)%mnistTrain.length)+1)%100 == 0){
						String shape = "" + network[i].inputSize;
						for(int j = 0; j < network[i].hiddenSize.length; j++){
							shape += " " + network[i].hiddenSize[j];
						}
						shape += " " + network[i].outputSize;
						System.out.println("Networks: " + network.length);
						log.info("Epoch " + epoch + ", Batch " + ((num/batchSize)%mnistTrain.length+1) + ": " + accuracy + ";  Network Shape: " + shape);
					}
					//set limits for adding layers or neurons
					newNetworks.add(new GrowingNN(network[i]));
					for(int j = 0; j < neuronCheck[i].length; j++){
						double sum = 0;
						int count = 1;
						Matrix abs = abs(g[count]);
						for(int k = 0; k < g[count].numRows(); k++){
							for(int l = 0; l < g[count].numColumns(); l++){
								sum += abs.get(k, l);
							}
						}
						count++;
						abs = abs(g[count]);
						for(int k = 0; k < g[count].numRows(); k++){
							for(int l = 0; l < g[count].numColumns(); l++){
								sum += abs.get(k, l);
							}
						}
						double avg = sum/(double)(g[count-1].numRows()*g[count-1].numColumns() + g[count].numRows()*g[count].numColumns());
						if(neuronCheck[i][j] == 0){
							neuronCheck[i][j] = avg;
						}else if(avg/neuronCheck[i][j] < 1){
							if(avg / neuronCheck[i][j] > .5){
								if(network[i].hiddenSize.length <= j){
									System.out.println(network[i].hiddenSize.length);
									System.out.println(neuronCheck[i].length);
								}
								network[i].addNeuron(j+2);
								reset = true;
							}else{
								neuronCheck[i][j] = avg;
							}
						}
					}
					Matrix one = new DenseMatrix(g[g.length-1].numColumns(), 1);
					for(int j = 0; j < one.numRows(); j++){
						one.set(j, 0, 1);
					}
					Matrix avgError = new DenseMatrix(1, 1);
					avgError = abs(g[g.length-1]).mult(1/(double)g[g.length-1].numColumns(), one, avgError);
					if(layerCheck[i] == 0){
						layerCheck[i] = avgError.get(0, 0);
					}else if(avgError.get(0, 0)/layerCheck[i] < 1){
						if(avgError.get(0, 0) / layerCheck[i] > .995){
							network[i].addLayer();
							reset = true;
						}else{
							layerCheck[i] = avgError.get(0, 0);
						}
					}
					if(reset){
						reset = false;
						fullReset = true;
						layerCheck[i] = 0;
						neuronCheck[i] = new double[network[i].hiddenLayer.length];
						for(int j = 0; j < neuronCheck[i].length; j++){
							neuronCheck[i][j] = 0;
						}
						//make sure new network isn't same architecture as existing network
						boolean same = true;
						for(int j = i; j < network.length; j++){
							if(network[i].hiddenLayer.length == network[j].hiddenLayer.length){
								for(int k = 0; k < network[i].hiddenLayer.length; k++){
									if(network[i].hiddenSize[k] != network[j].hiddenSize[k]){
										same = false;
									}
								}
							}else{
								same = false;
							}
						}
						for(int j = 0; j < newNetworks.size(); j++){
							if(network[i].hiddenLayer.length == newNetworks.get(j).hiddenLayer.length){
								for(int k = 0; k < network[i].hiddenLayer.length; k++){
									if(network[i].hiddenSize[k] != newNetworks.get(j).hiddenSize[k]){
										same = false;
									}
								}
							}else{
								same = false;
							}
						}
						if(!same){
							newNetworks.add(new GrowingNN(network[i]));
						}
					}	
				}
				//cycle through new networks, deleting old networks if there are bigger ones that are better
				Vector<Double> accuracy = new Vector<Double>();
				for(int i = 0; i < newNetworks.size(); i++){
					accuracy.add(newNetworks.get(i).verify(verifyImage, verifyLabelNum));
				}
				Vector<Integer> remove = new Vector<Integer>();
				for(int i = 0; i < newNetworks.size(); i++){
					for(int j = 0; j < newNetworks.size(); j++){
						if(i != j){
							boolean sameOrSmaller = true;
							if(newNetworks.get(i).hiddenSize.length > newNetworks.get(j).hiddenSize.length){
								sameOrSmaller = false;
							}else if(newNetworks.get(i).hiddenSize.length == newNetworks.get(j).hiddenSize.length){
								for(int k = 0; k < newNetworks.get(i).hiddenSize.length; k++){
									if(newNetworks.get(i).hiddenSize[k] > newNetworks.get(j).hiddenSize[k]){
										sameOrSmaller = false;
									}
								}
							}
							if(sameOrSmaller){
								//check fitness
								//if old fitness worse than new fitness, delete old network
								if(accuracy.get(i) < accuracy.get(j)){
									if(!remove.contains(i)){
										remove.add(i);
									}
								}
							}
						}
					}
				}
				for(int i = remove.size() - 1; i >= 0; i--){
					newNetworks.remove((int)remove.get(i));
					accuracy.remove((int)remove.get(i));
				}
				if(newNetworks.size() >= 100){
					remove.clear();
					for(int i = 0; i < newNetworks.size() - LOW_NUMBER_OF_NETWORKS; i++){
						int worst = -1;
						for(int j = 0; j < newNetworks.size(); j++){
							if(!remove.contains(j)){
								if(worst == -1){
									worst = j;
								}else{
									if(accuracy.get(j) < accuracy.get(worst)){
										worst = j;
									}
								}
							}
						}
						remove.add(worst);
					}
					int[] removeArray = new int[remove.size()];
					for(int i = 0; i < remove.size(); i++){
						removeArray[i] = remove.get(i);
					}
					Arrays.sort(removeArray);
					for(int i = removeArray.length - 1; i >= 0; i--){
						newNetworks.remove(removeArray[i]);
						accuracy.remove(removeArray[i]);
					}
				}
				remove.clear();
				accuracy.clear();
				network = newNetworks.toArray(new GrowingNN[0]);
				newNetworks.clear();
				if(fullReset){
					fullReset = false;
					layerCheck = new double[network.length];
					neuronCheck = new double[network.length][];
					for(int i = 0; i < network.length; i++){
						layerCheck[i] = 0;
						neuronCheck[i] = new double[network[i].hiddenSize.length];
						for(int j = 0; j < neuronCheck[i].length; j++){
							neuronCheck[i][j] = 0;
						}
					}
				}
			}
		}
    }
    
    public void test(String labelFilename, String imageFilename) throws IOException{
        DigitImageLoadingService mnistTestImport = new DigitImageLoadingService(labelFilename, imageFilename);
		DigitImage[] mnistTest = mnistTestImport.loadDigitImages().toArray(new DigitImage[0]);
		double[][] image = new double[mnistTest.length][];
		int[] label = new int[mnistTest.length];
		for(int i = 0; i < mnistTest.length; i++){
			image[i] = mnistTest[i].getData();
			label[i] = mnistTest[i].getLabel();
		}
		double[] accuracy = new double[network.length];
		int best = 0;
		for(int i = 0; i < accuracy.length; i++){
			accuracy[i] = network[i].verify(image, label);
			if(accuracy[i] > accuracy[best]){
				best = i;
			}
		}
		//log accuracy
		log.info("Test: " + accuracy[best]);
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
    
    public Matrix abs(Matrix matrix){
    	Matrix newMatrix = new DenseMatrix(matrix.numRows(), matrix.numColumns());
    	for(int i = 0; i < matrix.numRows(); i++){
    		for(int j = 0; j < matrix.numColumns(); j++){
    			newMatrix.set(i, j, Math.abs(matrix.get(i, j)));
    		}
    	}
    	return newMatrix;
    }
}
