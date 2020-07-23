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

import java.util.*;

import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.*;
import org.encog.ml.data.basic.*;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.*;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.*;
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing;
import org.encog.neural.networks.training.cross.NetworkFold;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.pattern.JordanPattern;

import tools.StatUtils;
import de.uos.fmt.musitech.utility.math.MyMath;

/**
 * Create and test NN sequence predictors.
 * 
 * @author Tillman
 * 
 */
public class ElmanSequence {

	static BasicNetwork createElmanNetwork() {
		// construct an Elman type network
		ModElmanPattern pattern = new ModElmanPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setActivationFunction2(new ActivationSoftMax());
		pattern.setInputNeurons(25);
		pattern.addHiddenLayer(5);
		pattern.setOutputNeurons(25);
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
	
	static BasicNetwork createFeedforwardNetwork1() {
		// construct a feedforward type network
		FeedForwardPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setActivationOutput(new ActivationSoftMax());
		pattern.setInputNeurons(25);
		pattern.addHiddenLayer(10);
		pattern.setOutputNeurons(25);
		return (BasicNetwork)pattern.generate();
	}

	static BasicNetwork createFeedforwardNetwork2() {
		// construct a feedforward type network
		FeedForwardPattern pattern = new FeedForwardPattern();
		pattern.setActivationFunction(new ActivationSigmoid());
		pattern.setActivationOutput(new ActivationSoftMax());
		pattern.setInputNeurons(50);
		pattern.addHiddenLayer(10);
		pattern.setOutputNeurons(25);
		return (BasicNetwork)pattern.generate();
	}



	public static void main1(final String args[]) {
		
		final TemporalXOR temp = new TemporalXOR();
		final MLDataSet trainingSet = temp.generateSin(4500); 
		final MLDataSet trainingSet2 = temp.generateSin2(4500); 

		final BasicNetwork elmanNetwork = ElmanSequence.createElmanNetwork();
		final BasicNetwork jordanNetwork = ElmanSequence.createElmanNetwork();
		final BasicNetwork ffwdNetwork = ElmanSequence.createFeedforwardNetwork2();
		
		MLData data = new BasicMLData(25);
		MLDataPair pair = new BasicMLDataPair(data);
		trainingSet.getRecord(0, pair);
		for (int i = 0; i < 3; i++) {
			System.out.println(data);
			data = elmanNetwork.compute(data);					
		}

		final double elmanError = ElmanSequence.trainNetwork("Elman", elmanNetwork, trainingSet);

		final double jordanError = ElmanSequence.trainNetwork("Jordan", jordanNetwork, trainingSet);

		
		final double ffwdError = ElmanSequence.trainNetwork("FeedFwd", ffwdNetwork, trainingSet2);

		System.out.println("Best error rate with Elman Network: " + elmanError);

		System.out.println("Best error rate with Jordan Network: " + jordanError);

		System.out.println("Best error rate with Feed Forward Network: " + ffwdError);

		int simMax = 10;
		
		System.out.println("Simulate Elman Network w ground truth: ");
		for (int i = 0; i < simMax; i++) {
			trainingSet.getRecord(i, pair);	
			data = pair.getInput();
			System.out.println(" In: "+data);
			MLData data2 = elmanNetwork.compute(data);					
			System.out.println("Out: "+data2);
		}
		System.out.println(data);

		System.out.println("Simulate Elman Network recursive: ");
		trainingSet.getRecord(0, pair);	
		data = pair.getInput();
		for (int i = 0; i < simMax; i++) {
			System.out.println(data);
			data = elmanNetwork.compute(data);					
		}
		System.out.println(data);

		System.out.println("Simulate Jordan Network w/ ground truth: ");
		for (int i = 0; i < simMax; i++) {
			trainingSet.getRecord(i, pair);	
			data = pair.getInput();
			System.out.println(" In: "+data);
			data = jordanNetwork.compute(data);	
			System.out.println("Out: "+data);
		}

		System.out.println("Simulate Jordan Network recursive: ");
		trainingSet.getRecord(0, pair);	
		data = pair.getInput();
		for (int i = 0; i < simMax; i++) {
			System.out.println(data);
			data = jordanNetwork.compute(data);					
		}
		System.out.println(data);

		System.out.println("Simulate FFwd Network w/ ground truth: ");
		for (int i = 0; i < simMax; i++) {
			trainingSet2.getRecord(i, pair);	
			data = pair.getInput();
			System.out.println(" In: "+data);
			data = ffwdNetwork.compute(data);					
			System.out.println("Out: "+data);
		}
		System.out.println(data);

		System.out.println("Simulate FFwd Network: ");
		trainingSet2.getRecord(0, pair);	
		data = pair.getInput();
		for (int i = 0; i < simMax; i++) {
			System.out.println(data);
			MLData data1 = ffwdNetwork.compute(data);
			double dataArray[] = data.getData();
			double dataArray2[] = Arrays.copyOfRange(dataArray, 25, 50);
			int j=0;
			for (; j < dataArray2.length; j++) {
				dataArray[j] = dataArray2[j];
			}
			double dataArray1[] = data1.getData();
			for (; j < 25+dataArray2.length; j++) {
				dataArray[j] = dataArray2[j-25];
			}
			data.setData(dataArray);
		}
		System.out.println(data);

		Encog.getInstance().shutdown();
	}
	
	public static void main(final String args[]) {
		
		final TemporalXOR temp = new TemporalXOR();
		final MLDataSet trainingSet = temp.generateSin(4500); 
		final MLDataSet trainingSet2 = temp.generateSin2(4500); 

		final BasicNetwork elmanNetwork = ElmanSequence.createElmanNetwork();
		final BasicNetwork jordanNetwork = ElmanSequence.createElmanNetwork();
		final BasicNetwork ffwdNetwork1 = ElmanSequence.createFeedforwardNetwork1();
		final BasicNetwork ffwdNetwork2 = ElmanSequence.createFeedforwardNetwork2();

		int k = 5;
		
		trainTestCV("Elman", trainingSet, elmanNetwork, k);
		trainTestCV("ffw1",trainingSet,ffwdNetwork1,k);
		trainTestCV("ffw2",trainingSet2,ffwdNetwork2,k);


		Encog.getInstance().shutdown();
	}

	static void trainTestCV(String name, final MLDataSet trainingSet, final BasicNetwork network, int k) {
		// create test and training folds 
		MLDataSet[] testFolds = splitTrainingSetLinear(trainingSet, k);
		MLDataSet[] trainFolds = new MLDataSet[k];
		for (int i = 0; i < testFolds.length; i++) {
			trainFolds[i] = creatTrainingSet(testFolds, i);
		}

		System.out.println("Testing " + name);
		NetworkFold[] nwf = trainCV(network, trainFolds);
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

	public static MLDataSet creatTrainingSet(MLDataSet folds[], int exclude){
		MLDataSet merged = new BasicMLDataSet();
		for (int i = 0; i < folds.length; i++) {
			if(i == exclude)
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
	public static MLDataSet creatTrainingSet(MLDataSet folds[], List<Integer> exclude){
		MLDataSet merged = new BasicMLDataSet();
		for (int i = 0; i < folds.length; i++) {
			if(exclude.contains(i))
				continue; // skip test and validation folds
			long len = folds[i].getRecordCount();
			for (long j = 0; j < len; j++) {
				MLDataPair pair = new BasicMLDataPair(null);
				folds[i].getRecord(j, pair); 
				merged.add(pair); 
			}
		}
		return merged;
	}
	
	public static NetworkFold[] trainCV(BasicNetwork network1, MLDataSet folds[]){
		FlatNetwork network = network1.getFlat();
		NetworkFold nwf[] = new NetworkFold[folds.length];
		for (int i = 0; i < folds.length; i++) {
			nwf[i] = new NetworkFold(network);
			nwf[i].copyFromNetwork(network);
		}
		for (int i = 0; i < folds.length; i++) {
			System.out.println("training fold "+i);
			nwf[i].copyToNetwork(network);
			trainNetwork("fold "+i, network1, folds[i]);			
			nwf[i].copyFromNetwork(network);
		}
		return nwf;
	}
	
	public static double[] evalCV(BasicNetwork network1, NetworkFold[] nwf, MLDataSet folds[]){
		FlatNetwork network = network1.getFlat();
		double[] errors = new double[folds.length];
		for (int i = 0; i < folds.length; i++) {
			nwf[i].copyToNetwork(network);
			errors[i] = network.calculateError(folds[i]);			
		}
		return errors;
	}
	

	public static double trainNetwork(final String what,
			final BasicNetwork network, final MLDataSet trainingSet) {
		// train the neural network
		CalculateScore score = new TrainingSetScore(trainingSet);
		final MLTrain trainAlt = new NeuralSimulatedAnnealing(
				network, score, 5, 2, 10);

//		final MLTrain trainMain = new Backpropagation(network, trainingSet,0.00065, 0.4);
		final MLTrain trainMain = new ResilientPropagation(network, trainingSet);

		final StopTrainingStrategy stop = new StopTrainingStrategy();
		trainMain.addStrategy(new Greedy());
//		trainMain.addStrategy(new HybridStrategy(trainAlt));
		trainMain.addStrategy(stop);

		int epoch = 0;
		while (!stop.shouldStop()) {
			trainMain.iteration();
			if(epoch%100 == 0)
				System.out.println("Training " + what + ", Epoch #" + epoch
					+ " Error:" + trainMain.getError());
			epoch++;
			if(epoch == 10000)
				break;
		}
		System.out.println("Training " + what + ", finalError:" + trainMain.getError());
		return trainMain.getError();
	}
}