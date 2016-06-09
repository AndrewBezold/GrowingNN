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
public class MNIST2 
{
	public GrowingNN network;
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
	    	MNIST2 mnist = new MNIST2();
	    	mnist.network = new GrowingNN(784, 3, 10);
	        	mnist.train(mnistTrainLabelFilename, mnistTrainImageFilename);
	        	mnist.test(mnistTestLabelFilename, mnistTestImageFilename);
	        	//output network
	        	mnist.network.output("GrowingNN" + time + ".network");
        } catch (IOException e) {
			e.printStackTrace();
		}
        
    }
    
    public void train(String labelFilename, String imageFilename) throws IOException{
    	int newLayerNeurons = 0;
    	
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
		int numEpochs = 10;
		int iterator = 0;
		int epoch = 1;
		boolean endOfEpoch = false;
		double startError = 0;
		double currentError = 0;
		DigitImage[] batch;
		for(int num = 0; num < mnistTrain.length * numEpochs + batchSize; num += batchSize){
			int thisSize;
			if(num + batchSize > mnistTrain.length * numEpochs){
				thisSize = mnistTrain.length * numEpochs - num;
			}else{
				thisSize = batchSize;
			}
			if(thisSize > 0){
				//System.out.println("Batch " + ((num/batchSize) + 1));
				batch = new DigitImage[thisSize];
				for(int i = 0; i < thisSize; i++){
					if(iterator >= mnistTrain.length){
						shuffle(mnistTrain);
						iterator = 0;
						epoch++;
						endOfEpoch = true;
					}
					batch[i] = mnistTrain[iterator];
					iterator++;
				}
				//grow network if finished epoch last minibatch
				if(endOfEpoch){
					System.out.println("End Epoch Test");
					endOfEpoch = false;
					//add neurons based on current error and error decrease over batch
					//higher error = more neurons
					//higher decrease = less neurons
					//number of added neurons decreases the deeper you go
					//add layer based on idk.  If number of neurons added at said layer passes minimum threshold?
					//if so, keep track of number of neurons to be added until it passes the minimum threshold
					int numNeurons;
					
					//500 new neurons at 10 error and 0 error decrease
					//0 new neurons at 0 error
					//new neurons change proportional to newError/oldError?
					numNeurons = (int) (500*(1/(1+Math.exp(0.5*((currentError-startError)-5))))*(1/(1+Math.exp(-0.5*((currentError)+5)))));
					System.out.println(numNeurons + " " + currentError + " " + startError);
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
					startError = 0;
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
				Matrix[] g = network.adam(image, label, network.learningRate, network.b1, network.b2, network.e);
				//verify against trained images
				double[] accuracy = network.verify(verifyImage, verifyLabelNum);
				if(startError == 0){
					startError = accuracy[1];
				}
				currentError = accuracy[1];
				g = network.gradients(new DenseMatrix(verifyImage), new DenseMatrix(verifyLabel));
				//log accuracy
				if((((num/batchSize)%mnistTrain.length)+1)%100 == 0){
					String shape = "" + network.inputSize;
					for(int j = 0; j < network.hiddenSize.length; j++){
						shape += " " + network.hiddenSize[j];
					}
					shape += " " + network.outputSize;
					log.info("Epoch " + epoch + ", Batch " + (((num/batchSize)%mnistTrain.length)+1) + ": " + accuracy[0] + ";  Network Shape: " + shape);
				}
				//set limits for adding layers or neurons
				
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
		double accuracy = network.verify(image, label)[0];
		//log accuracy
		log.info("Test: " + accuracy);
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
