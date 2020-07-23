package models;

import org.encog.ml.data.*;
import org.encog.ml.data.folded.FoldedDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.cross.CrossValidationKFold;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class CV_test {

	public class CV_results {
		
	}

	private static final double MAX_ERROR = 0.1;

	public CrossValidationKFold c_val(MLDataSet trainingData, BasicNetwork network, int folds) {
		FoldedDataSet folded = new FoldedDataSet(trainingData);
		Train train = new ResilientPropagation(network, folded);
		CrossValidationKFold trainFolded = new CrossValidationKFold(train,folds);

		int epoch = 1;

		do {
    		trainFolded.iteration();
    		System.out.println("Epoch #" + epoch + " Error:" + trainFolded.getError());
    		epoch++;
		} while (trainFolded.getError() > MAX_ERROR);
		return trainFolded;
	}

	private static MLData getData() {
		// TODO Auto-generated method stub
		return null;
	}
	
	
}
