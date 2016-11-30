package com.bezold.nn.growing_nn;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.ThreadContext;

import com.bezold.mnist.DigitImage;
import com.bezold.mnist.DigitImageLoadingService;

public class MNIST 
{
	public GrowingNN network;
	private static final Logger log = LogManager.getLogger();
	
    public static void main( String[] args )
    {
    	String time = LocalDateTime.now().format(DateTimeFormatter.ofPattern("uuuu-MM-dd'T'HH_mm_ss_SSS"));
    	try {
	    	ThreadContext.put("logFileName", time);
	    	String mnistTrainImageFilename = "/home/beez/mnist/train-images.idx3-ubyte";
	    	String mnistTrainLabelFilename = "/home/beez/mnist/train-labels.idx1-ubyte";
	    	String mnistTestImageFilename = "/home/beez/mnist/t10k-images.idx3-ubyte";
	    	String mnistTestLabelFilename = "/home/beez/mnist/t10k-labels.idx1-ubyte";
	    	MNIST mnist = new MNIST();
	    	mnist.network = new GrowingNN(784, new int[]{3}, 10);
	        mnist.train(mnistTrainLabelFilename, mnistTrainImageFilename);
	        mnist.test(mnistTestLabelFilename, mnistTestImageFilename);
	        //output network
	        mnist.network.output("GrowingNN" + time + ".network");
        } catch (IOException e) {
			e.printStackTrace();
		}
        
    }
    
    public void train(String labelFilename, String imageFilename) throws IOException{
    	
        DigitImageLoadingService mnistTrainImport = new DigitImageLoadingService(labelFilename, imageFilename);
		DigitImage[] mnistFull = mnistTrainImport.loadDigitImages().toArray(new DigitImage[0]);
		DigitImage[] mnistTrain = new DigitImage[55000];
		DigitImage[] mnistVerify = new DigitImage[5000];
		
		shuffle(mnistFull);
		for(int i = 0; i < mnistFull.length; i++){
			if(i < mnistTrain.length){
				mnistTrain[i] = mnistFull[i];
			}else{
				mnistVerify[i-mnistTrain.length] = mnistFull[i];
			}
		}
		float[][] trainImage = new float[mnistTrain.length][784];
		float[][] trainLabel = new float[mnistTrain.length][10];
		int[] trainLabelNum = new int[mnistTrain.length];
		float[][] verifyImage = new float[mnistVerify.length][784];
		float[][] verifyLabel = new float[mnistVerify.length][10];
		int[] verifyLabelNum = new int[mnistVerify.length];
		int[] maxn = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		for(int i = 0; i < mnistTrain.length; i++){
			trainImage[i] = mnistTrain[i].getData();
			trainLabelNum[i] = mnistTrain[i].getLabel();
			for(int j = 0; j < 10; j++){
				if(j==trainLabelNum[i]){
					maxn[j]++;
					trainLabel[i][j] = 1;
				}else{
					trainLabel[i][j] = 0;
				}
			}
			
		}
		System.out.println("0: " + maxn[0] + " 1: " + maxn[1] + " 2: " + maxn[2] + " 3: " + maxn[3] + " 4: " + maxn[4] + " 5: " + maxn[5] + " 6: " + maxn[6] + " 7: " + maxn[7]+ " 8: " + maxn[8] + " 9: " + maxn[9]);
		int[] maxv = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		for(int i = 0; i < mnistVerify.length; i++){
			verifyImage[i] = mnistVerify[i].getData();
			verifyLabelNum[i] = mnistVerify[i].getLabel();
			for(int j = 0; j < 10; j++){
				if(j==verifyLabelNum[i]){
					maxv[j]++;
					verifyLabel[i][j] = 1;
				}else{
					verifyLabel[i][j] = 0;
				}
			}
		}
		System.out.println("0: " + maxv[0] + " 1: " + maxv[1] + " 2: " + maxv[2] + " 3: " + maxv[3] + " 4: " + maxv[4] + " 5: " + maxv[5] + " 6: " + maxv[6] + " 7: " + maxv[7]+ " 8: " + maxv[8] + " 9: " + maxv[9]);
		
		int batchSize = 50;
		int numEpochs = 50;
		network.grow(trainImage, trainLabel, verifyImage, verifyLabel, numEpochs, batchSize, log);
		System.out.println(network.verifySize());

    }
    
    public void test(String labelFilename, String imageFilename) throws IOException{
        DigitImageLoadingService mnistTestImport = new DigitImageLoadingService(labelFilename, imageFilename);
		DigitImage[] mnistTest = mnistTestImport.loadDigitImages().toArray(new DigitImage[0]);
		float[][] image = new float[mnistTest.length][];
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
