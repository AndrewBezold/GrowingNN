package com.bezold.mnist;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrix;

public class Test {

	public static void main(String[] args){
		Matrix matrix = new DenseMatrix(1, 1);
		double num = .000001;
		matrix.set(0, 0, num);
		double getNum = matrix.get(0, 0);
		System.out.println(num + " " + matrix + " " + getNum);
	}
}
