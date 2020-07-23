package models;
/*
 * Encog(tm) Java Examples v3.2
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-examples
 *
 * Copyright 2008-2013 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */

import java.io.*;
import java.util.*;

import org.apache.commons.lang3.ArrayUtils;
import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.*;
import org.encog.ml.data.basic.*;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.*;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.*;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.pattern.JordanPattern;

import tools.StatUtils;
import data.DataReader;
import de.uos.fmt.musitech.utility.math.MyMath;

/**
 * Create and test NN sequence predictors.
 * 
 * @author tweyde
 *
 */
public class SequenceModel implements ISequenceModel {
	
	public static final int MIN = 30;
	public static final int MAX = 80;

	static BasicNetwork createElmanNetwork(int hidden,int out) {
		// construct an Elman type network
		ModElmanPattern pattern = new ModElmanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setActivationFunction2(new ActivationSoftMax());
		pattern.setInputNeurons(4);
		pattern.addHiddenLayer(hidden);
		pattern.setOutputNeurons(out);
		return (BasicNetwork)pattern.generate();
	}

	static BasicNetwork createJordanNetwork() {
		// construct an Jordan type network
		JordanPattern pattern = new JordanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setInputNeurons(5);
		pattern.addHiddenLayer(50);
		pattern.setOutputNeurons(25);
		return (BasicNetwork)pattern.generate();
	}
	
	static BasicNetwork createFeedforwardNetwork1(int out) {
		// construct a feedforward type network
		FeedForwardPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setActivationOutput(new ActivationLinear());
		pattern.setInputNeurons(4);
		pattern.addHiddenLayer(8);
		pattern.setOutputNeurons(out);
		return (BasicNetwork)pattern.generate();
	}

	static BasicNetwork createFeedforwardNetwork2() {
		// construct a feedforward type network
		FeedForwardPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
//		pattern.setActivationOutput(new ActivationSoftMax());
		pattern.setInputNeurons(8);
		pattern.addHiddenLayer(8);
		pattern.setOutputNeurons(1);
		return (BasicNetwork)pattern.generate();
	}

	static BasicNetwork createFeedforwardNetwork(int context, int hidden, int out) {
		// construct a feedforward type network
		FeedForwardPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setActivationOutput(new ActivationSoftMax());
		pattern.setInputNeurons(context*4);
		pattern.addHiddenLayer(hidden);
		pattern.setOutputNeurons(out);
		return (BasicNetwork)pattern.generate();
	}

	
	/**
	 * Create an MLDataSet from a map of piece names to lists of voices (represented as lists of feature vectors). 
	 * @param map The map from piece names to lists of voices. 
	 * @param context The size of the context, i.e. number of notes, in the input. 
	 * @return The MLDataSet with as many pairs as there are notes in all voices of all pieces. 
	 */
	public static MLDataSet[] toMLDataSet(Map<String,List<List<List<Double>>>> map, int context){
		List<MLDataSet> list = new ArrayList<MLDataSet>();
		Collection<List<List<List<Double>>>> pieces = map.values();
		for(List<List<List<Double>>> voices: pieces){
			MLDataSet allVoiceDataSet = voices2data(voices, context); 
			list.add(allVoiceDataSet); 
		} 
		MLDataSet[] datasets = list.toArray(new MLDataSet[list.size()]);
		int in = datasets[0].getInputSize();
		int t = datasets[0].getIdealSize();
		System.out.print("Input size: " + in + "target size: "+t+"\nFold sizes: ");
		double cnts[] = new double[datasets.length];
		for (int i = 0; i < datasets.length; i++) {
			long c = datasets[i].getRecordCount();
			System.out.print(c + ", ");
			cnts[i] = c;
		}
		System.out.println("Avg: "+StatUtils.mean(cnts));
		return datasets;
	}

	
	/**
	 * Generate an MLDataSet from a list of voices encoded as note-wise feature vectors. 
	 * The notes are the target vectors and the <i>context</i> previous notes constitute the input vector. 
	 * @param voices A list of feature vectors (lists of doubles). 
	 * @param context The context size (number of notes). 
	 * @return The data set, containing 1 pair per note. 
	 */
	static MLDataSet voices2data(List<List<List<Double>>> voices, int context) {
		MLDataSet allVoiceDataSet = null;	
		for(List<List<Double>> voice: voices){ 
			MLDataSet vdat = voice2MLData(voice, context); 
			if(allVoiceDataSet == null)
				allVoiceDataSet = vdat;
			else
				for (MLDataPair mlDataPair : vdat) {
					allVoiceDataSet.add(mlDataPair.getInput(), mlDataPair.getIdeal());
				}
		}
		return allVoiceDataSet;
	}

	private static MLDataSet voice2MLData(List<List<Double>> voice, int context) {
		int fSize = voice.get(0).size();
		double ideal[][] = new double[voice.size()][];
		double zeros[] = new double[fSize];
		double input[][] = new double[voice.size()][fSize*context];
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		for(int i = 0; i<voice.size(); i++){
//			for (int j = 0; j < fSize; j++) {
//			ideal[i][0] = voice.get(i).get(0);	
			ideal[i] = toOneHot(voice.get(i).get(0));	
			double val = voice.get(i).get(0);
			if(min>val)
				min = val;
			if(max<val)
				max = val;
//			}
			if(val > MAX)
				throw new RuntimeException("MAX is too low, needs to be at least "+val);
			if(val < MIN)
				throw new RuntimeException("MIN is too high, needs to be at most "+val);
			if(i>1) // set previous item as input
				for (int j = 0; j < fSize; j++) {
					input[i][j] = voice.get(i-1).get(j);				
				}
			else
				input[i] = zeros;
			int j = 1;
			while(j<context){ // add more context, if specified
				if(i>=j)
					input[i] = ArrayUtils.addAll(input[i], ArrayUtils.toPrimitive(voice.get(i-j).toArray(new Double[fSize])));
				else
					input[i] = ArrayUtils.addAll(input[i], zeros);
				j++;
			}
		}
//		System.out.println("Min: "+min+", max: "+max);
		MLDataSet dataSet = new BasicMLDataSet(input,ideal);
		return dataSet;
	}
	
	static double[] toOneHot(double val){
		double[] encoded = new double[MAX-MIN+1];
		for (int i = 0; i < encoded.length; i++) {
			encoded[i] = 0;
		}
		int index = (int)(val-MIN);
		//System.out.println(val-min-index);
		//System.out.println(index);
		encoded[index] = 1;
		return encoded;
	}
	
	/**
	 * Expect a list as: pitch, duration (whole notes), pitchDifference, ioi (whole notes).
	 * @param subMelody
	 * @param voiceNum
	 * @return
	 */
	@Override
	public double modelProbability(List<List<Double>> subMelody, int voiceNum){
		if(subMelody.size() == 0)
			return -1;
		int fSize = subMelody.get(0).size();
		double output[] = new double[out];
		double input[] = new double[context*fSize];
		int last = subMelody.size()-1;
		int k = 0;
		// TODO check i not negative, use default encoding
		for (int i = last-context; i < last; i++) {
			for (int j = 0; j < fSize; j++) {
				if(i>0)
					input[k] = subMelody.get(i).get(j);
				else 
					input[k] = 0;
				k++;
			}
		}
		network.compute(input,output);
		return output[(int)(subMelody.get(last).get(0).doubleValue())-MIN];
	}
	
	BasicNetwork network;
	int context = 6;
	int out = MAX - MIN + 1;
	
	@Override
	public void trainModel(List<List<List<Double>>> melodyList){
		MLDataSet trainingSet = voices2data(melodyList, context);
		network = createFeedforwardNetwork(6, 10, out);
		double error = trainNetwork("", network, trainingSet);
		System.out.println("Error after training = " + error);
	}

	@Override
	public void saveModel(File f){
		double encoded[] = new double[network.encodedArrayLength()];
		network.encodeToArray(encoded);
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(f));
			oos.writeObject(encoded);
			oos.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void loadModel(File f){
		double encoded[];
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f));
			encoded = (double[]) ois.readObject();
			ois.close();
			if(network == null)
				network = new BasicNetwork();
			network.decodeFromArray(encoded);		
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public Object getModel(){
		Weights weights = new Weights();
		double encoded[] = new double[network.encodedArrayLength()];
		network.encodeToArray(encoded);
		weights.setWeights(encoded);
		return weights;
	}
	
	public static class Weights{
		double weights[];

		/**
		 * @return the weights
		 */
		public double[] getWeights() {
			return weights;
		}

		/**
		 * @param weights the weights to set
		 */
		public void setWeights(double[] weights) {
			this.weights = weights;
		}
	}

	public void setModel(Object model){
//		if(network == null)
//			network = createFeedforwardNetwork(6, 10, out);
////			network = new BasicNetwork();
		if(model != null){
			network.decodeFromArray(((Weights)model).getWeights());
		} else
			System.err.println("WARNING: model is null");
	}
	
	public static void main(final String args[]) {
		
		SequenceModel sm = new SequenceModel();
		DataReader dr = DataReader. getInstance();
		Collection<List<List<List<Double>>>> col = dr.tvtMap.values();
		List<List<List<Double>>> voices = col.iterator().next();
		sm.trainModel(voices);
//		File f = new File("temp.ser");
//		sm.saveModel(f);
//		sm.loadModel(f);
		voices = col.iterator().next();
		sm.modelProbability(voices.get(0), 1);

//		Map<String,List<List<List<Double>>>> map = dr.tvtMap;
//		MLDataSet dataFolds1[] = toMLDataSet(map, 1);
////		MLDataSet dataFolds2[] = toMLDataSet(map, 2);
//		MLDataSet dataFolds3[] = toMLDataSet(map, 8);
//		
//		
//		
//		int out = MAX - MIN + 1;
//
//		final BasicNetwork elmanNetwork = SequenceModel.createElmanNetwork(8, out);
////		final BasicNetwork jordanNetwork = SequenceModelTest.createElmanNetwork();
////		final BasicNetwork ffwdNetwork1 = SequenceModelTest.createFeedforwardNetwork1();
////		final BasicNetwork ffwdNetwork2 = SequenceModelTest.createFeedforwardNetwork2();
//		final BasicNetwork ffwdNetwork3 = SequenceModel.createFeedforwardNetwork(8,16, out);
//		
//		trainTestCV("Elman", dataFolds1, elmanNetwork);
////		trainTestCV("ffw1", dataFolds1, ffwdNetwork1);
////		trainTestCV("ffw2", dataFolds2, ffwdNetwork2);
//		trainTestCV("ffw3", dataFolds3, ffwdNetwork3);
//
		Encog.getInstance().shutdown();
	}

	static void trainTestCV(String name, final MLDataSet trainingSet, final BasicNetwork network, int k) {
		// create test and training folds 
		MLDataSet[] testFolds = splitTrainingSetLinear(trainingSet, k);
		trainTestCV(name, testFolds, network);
	}
	
	static void trainTestCV(String name, final MLDataSet[] testFolds, final BasicNetwork network) {
			// create test and training folds 
		int k = testFolds.length;
		MLDataSet[] trainFolds = new MLDataSet[k];
		for (int i = 0; i < testFolds.length; i++) {
			trainFolds[i] = createTrainingSet(testFolds, i);
		}

		System.out.println("Testing " + name);
		BasicNetwork[] nwf = trainCV(network, trainFolds);
		double[] errors = evalCV(network, nwf, trainFolds);
		System.out.println(name+", train, mean: "+MyMath.mean(errors) + " std: " + StatUtils.standardDeviationSample(errors));
		errors = evalCV(network, nwf, testFolds);
		System.out.println(name+", test, mean: "+MyMath.mean(errors) + " std: " + StatUtils.standardDeviationSample(errors));
	}

	public static MLDataSet[] splitTrainingSetLinear(MLDataSet data, int num){
		MLDataSet folds[] = new MLDataSet[num];
		for (int i = 0; i < folds.length; i++) {
			folds[i] = new BasicMLDataSet();
		}
		long len = data.getRecordCount();
		long foldSize = len/num;
		for (long i = 0; i < len; i++) {
			MLDataPair pair = new BasicMLDataPair(new BasicMLData(0),new BasicMLData(0));
			data.getRecord(i, pair); 
			folds[(int) (i/foldSize)].add(pair); 
		}
		for (int i = 0; i < folds.length; i++) {
			System.out.println("fold" + i + ", size: " + folds[i].getRecordCount());
		}
		return folds;
	}

	public static MLDataSet createTrainingSet(MLDataSet folds[], int exclude){
		List<Integer> list = new ArrayList<Integer>();
		list.add(exclude);
		return createTrainingSet(folds, list);
	}
	
	public static MLDataSet createTrainingSet(MLDataSet folds[], List<Integer> exclude){
		MLDataSet merged = new BasicMLDataSet();
		for (int i = 0; i < folds.length; i++) {
			if(exclude.contains(i))
				continue; // skip test and validation folds
			long len = folds[i].getRecordCount();
			for (long j = 0; j < len; j++) {
				MLDataPair pair = new BasicMLDataPair(new BasicMLData(0),new BasicMLData(0));
				folds[i].getRecord(j, pair); 
				merged.add(pair); 
			}
		}
		return merged;
	}
	
	public static BasicNetwork[] trainCV(BasicNetwork network1, MLDataSet folds[]){
		BasicNetwork[] nwf = new BasicNetwork[folds.length];
		for (int i = 0; i < folds.length; i++) {
			nwf[i] = (BasicNetwork) network1.clone();
		}
		for (int i = 0; i < folds.length; i++) {
			//System.out.println("training fold "+i);
			trainNetwork("fold "+i, nwf[i], folds[i]);			
		}
		return nwf;
	}
	
	public static double[] evalCV(BasicNetwork network1, BasicNetwork[] nwf, MLDataSet folds[]){
		double[] errors = new double[folds.length];
		for (int i = 0; i < folds.length; i++) {
			errors[i] = nwf[i].calculateError(folds[i]);			
		}
		return errors;
	}
	

	public static double trainNetwork(final String what,
			final BasicNetwork network, final MLDataSet trainingSet) {
		// train the neural network
		CalculateScore score = new TrainingSetScore(trainingSet);
		final MLTrain trainAlt = new NeuralSimulatedAnnealing(
				network, score, 10, 2, 5);

//		final MLTrain trainMain = new Backpropagation(network, trainingSet,0.0001, 0.4);
//		final MLTrain trainMain = new ScaledConjugateGradient(network, trainingSet);
		final MLTrain trainMain = new ResilientPropagation(network, trainingSet);

		final StopTrainingStrategy stop = new StopTrainingStrategy();
		trainMain.addStrategy(new Greedy());
//		trainMain.addStrategy(new HybridStrategy(trainAlt));
		trainMain.addStrategy(stop);

		int epoch = 0;
		while (!stop.shouldStop()) {
			trainMain.iteration();
			if(epoch%200 == 0)
				System.out.println("Training " + what + ", Epoch #" + epoch
					+ " Error:" + trainMain.getError());
			epoch++;
			if(epoch == 1000)
				break;
		}
		System.out.println("Training " + what + ", final Error:" + trainMain.getError());
		System.out.println("Training " + what + ", finalError2:" + network.calculateError(trainingSet));
		return trainMain.getError();
	}

	@Override
	public void resetShortTermModel() {
		// TODO Auto-generated method stub
		
	}
}