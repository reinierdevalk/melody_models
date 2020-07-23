package models;

import java.io.File;
import java.util.List;

public interface ISequenceModel {
	
	/** 
	 * Get the probability of the last note in the provided list of items. 
	 * @param list The sequence up to here.
	 * @param voiceNum The number of the current voice. 
	 * @return The probability according to the model. 
	 */
	double modelProbability(List<List<Double>> list, int voiceNum); 

	/**
	 * Reset the short term model, normally at the beginning of a piece.
	 */
	void resetShortTermModel();

	/**
	 * Train the LTM
	 * @param melodyList
	 */
	void trainModel(List<List<List<Double>>> melodyList);

	/**
	 * Save the current LTM to a file
	 * @param f The File to save to.
	 */
	void saveModel(File f);
	
	/**
	 * Load an LTM from file.
	 * @param f
	 */
	void loadModel(File f);

}
