package n_grams;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import models.ISequenceModel;

import org.apache.commons.lang3.ArrayUtils;

import tools.ToolBox;
import de.uos.fmt.musitech.utility.obj.ObjectCopy;

/**
 * Simple Language Model with Laplace Smoothing and Simple Back-off
 * @author tweyde
 */
public class SimpleLM implements ISequenceModel, Serializable {
	
	public enum Type{LONG,SHORT,MIX}; 
	Type useSTM; 
	protected int N = Integer.MAX_VALUE; 
	public static final double invlog2 = 1/ Math.log(2);
	
	/**
	 * Create an new 
	 * @param argUseSTM
	 */
	public SimpleLM(Type argUseSTM){
		useSTM = argUseSTM;
	}
	
	public SimpleLM(Type argUseSTM, int argN){
		useSTM = argUseSTM;
		N = argN;
	}
	
	// serial#
	private static final long serialVersionUID = 2536367845478314674L;

	Double startSymbol = new Double(Double.NEGATIVE_INFINITY);
	List<Node<Double>> models = new ArrayList<Node<Double>>(); 
	Node<Double> seqTree = new Node<Double>(startSymbol,0);
	enum SMOOTHING {OFF, LAPLACE, BACKOFF, LPBACKOFF}  
	
	Node<Double> seqTreeST = new Node<Double>(startSymbol,0);


	public double crossEntropy(Double seq[]){
		double ce = 0;
		for (int i = 1; i <= seq.length; i++) {
			Double seq1[] = ArrayUtils.subarray(seq, 0, i);
			ce -= Math.log(seqTree.getBackoffProb(this, seq1)) * invlog2;
		}
		ce /= seq.length;
		return ce;
	}

	/**
	 * Update the tree with sequence data.
	 * @param seq the sequence of items to add to the tree.
	 */
	public void fillTree(Double seq[]) {
		for(int o = 0; o < seq.length; o++ )
			fillTree(seq, o);
	}
	
	/**
	 * Update the tree with sequence data, starting from an offset.
	 * @param seq the sequence of items to add to the tree.
	 * @param the offset to start from
	 */
	public void fillTree(Double seq[], int offset){
		int end = Math.min(offset+N, seq.length);
		for (int i = offset; i < seq.length; i++) {
			seqTree = seqTree.add(seq[i],1); 
		}
	}

	/**
	 * Add a sequence to the model from a given root.
	 * @param root The root node to start with. 
	 * @param seq The sequence to add.
	 */
	public void fillTree(Node<Double> root, Double seq[]){
		int end = Math.min(N, seq.length);
		for(int o = 0; o < end; o++ )
			fillTree(root, seq, o, end-o);
	}
	
	/**
	 * Adds a sequence from an offset to the model starting at root. 
	 * @param root The tree root. 
	 * @param seq The sequence.
	 * @param offset The offset to apply
	 */
	public void fillTree(Node<Double> root, Double seq[], int offset, int len){
		for (int i = offset; i < offset + len; i++) {
			root = root.add(seq[i],1); 
		}
	}

	public SimpleLM() {
		
	}
	
	public static class Node<T> implements Serializable{
		
		public Node(){			
		}

		T symbol;
		int count = 1;
		int depth = 0;
		Map<T,Node<T>> children = new HashMap<T,Node<T>>();
		
		public Node(T sym, int cnt){
			this.symbol = sym;
			this.count = cnt;
		}
		
		public Node<T> getChild(T s) {
			return children.get(s);
		}
		
		public Node<T> addChild(T s, int i) {
			Node<T> n = new Node<T>(s,i);
			n.depth = this.depth+1;
			children.put(n.symbol,n); 
			return n;
		}
		
		@Override
		public String toString(){
			return "Node- s: " + symbol + ", n: " + count + ", d: " + depth + ", c:" + children.size();
		}
		
		/**
		 * Add a symbol to a child to this node, creates a child if needed. 
		 * @param s The symbol to add.
		 * @param i 
		 * @return The child to which the symbol has been added. 
		 */
		public Node<T> add(T s, int i){
			Node<T> n = this.getChild(s);
			if(n == null){ // child doesn't exist, make new one. 
				return addChild(s,i);
			} else {	// else increment count of existing child. 
				n.count+=i;
				return n;
			}
		}
		
		/**
		 * Adds the counts of all children. 
		 * @return The sum of all childrens' counts.
		 */
		public int countSuccessors(){
			int c = 0;
			for (Node<T> n : children.values()) {
				c += n.count;
			}
			return c;
		}
		
		public double getProbability(SimpleLM model, T s){
			Node<T> chld = getChild(s);
			if(chld == null)
				return getUnobservedProb(model);
			int cnt = getChild(s).count;
			int sum = countSuccessors();
			return cnt / (double)sum;
		}

		
		/**
		 * Get the Laplace smoothed probability of symbol s at this node. 
		 * @param s The symbol
		 * @return The smoothed prob (sob+1)/(allobs+alphabetSize).
		 */
		public double getLpSmoothedProb(SimpleLM model, T s){
			Node<T> chd = getChild(s);
			int cnt = 1;
			if(chd != null)
				cnt += chd.count;
			int sum = model.seqTree.children.size() + countSuccessors();
			if(sum < minAlphaSize)
				sum = minAlphaSize;
			return cnt / (double)sum;
		}
		
		int minAlphaSize = 25;
		
		public double getUnobservedProb(SimpleLM model){
			int denom = Math.max(minAlphaSize, model.seqTree.children.size());
			return 1.0 / denom ;
		}
		
		/**
		 * Get the probabilities of the last symbol of this sequence for all context lengths.
		 * @param seq
		 * @return
		 */
		public double[] getProbs(SimpleLM model, T[] seq){
			double probs[] = 	new double[seq.length];
			int last = seq.length-1;
			for (int context = 1; context <= seq.length; context++) {
				Node<T> n = this;
				for(int i = seq.length-context; i<last;i++){
					n = n.getChild(seq[i]);
					if(n == null){
    					for(i=context-1;i<seq.length;i++) // outer loop will terminate after this.
    						probs[i] = getUnobservedProb(model); 
    					return probs;
					} 
				}
				//probs[context-1] = n.getProbability(seq[last]); 
				probs[context-1] = n.getLpSmoothedProb(model, seq[last]); 
			}
			return probs;
		}

		/**
		 * Get the n-gram probability of the last symbol of this sequence.
		 * @param seq
		 * @param n
		 * @return
		 */
		public double getProb(SimpleLM model, T[] seq, int context){
			int last = seq.length-1;
			Node<T> n = this;
			for(int i = seq.length-context; i<last;i++){
				n = n.getChild(seq[i]);
				if(n == null){
					return getUnobservedProb(model);
				} 
			}
			return n.getLpSmoothedProb(model,seq[last]); 
		}

		/**
		 * Get the simple backoff probability of the last symbol of this sequence.
		 * @param seq
		 * @return
		 */
		public double getBackoffProb(SimpleLM model, T[] seq){
			int last = seq.length-1;
			double lastProb = getUnobservedProb(model);
			for (int context = 1; context <= seq.length; context++) {
				Node<T> n = this;
				for(int i = seq.length-context; i<last;i++){
					n = n.getChild(seq[i]);
					if(n == null){
    					return lastProb;
					} 
				}
				lastProb = n.getLpSmoothedProb(model, seq[last]); 
			}
			return lastProb;
		}
		
		/**
		 * @return the symbol
		 */
		public T getSymbol() {
			return symbol;
		}

		/**
		 * @param symbol the symbol to set
		 */
		public void setSymbol(T symbol) {
			this.symbol = symbol;
		}

		/**
		 * @return the count
		 */
		public int getCount() {
			return count;
		}

		/**
		 * @param count the count to set
		 */
		public void setCount(int count) {
			this.count = count;
		}

		/**
		 * @return the depth
		 */
		public int getDepth() {
			return depth;
		}

		/**
		 * @param depth the depth to set
		 */
		public void setDepth(int depth) {
			this.depth = depth;
		}

		/**
		 * @return the children
		 */
		public Map<T, Node<T>> getChildren() {
			return children;
		}

		/**
		 * @param children the children to set
		 */
		public void setChildren(Map<T, Node<T>> children) {
			this.children = children;
		}



	}
	
	List<Double> sliceList(List<List<Double>> list, int num){
		List<Double> slice = new ArrayList<Double>(list.size());
		for (List<Double> vec : list) {
			slice.add(vec.get(num));
		}
		return slice;
	}

	List<List<Boolean>> lengthVoiceDone = new ArrayList<List<Boolean>>();
	
	@Override
	public double modelProbability(List<List<Double>> list, int voiceNum) {		

		if(list.size()==0)
			return 0;
		// slice list
		List<Double> relPitchSlice = sliceList(list,2);
		Double relPitchArray[] = relPitchSlice.toArray( new Double[relPitchSlice.size()] );

		// calculate LT probability
		double probLT = seqTree.getBackoffProb( this, relPitchArray );
		if(useSTM == Type.LONG){
			return probLT;
		}
		else {
    		// train ST model
    		int context = list.size()-1;
    		if( lengthVoiceDone.size() <= context){
    			List<Boolean> voiceDone = new ArrayList<Boolean>(5);
    			for (int j = 0; j < 5; j++) {
    				voiceDone.add(Boolean.FALSE);
    			}
    			lengthVoiceDone.add(voiceDone);
    		}
    		if(!lengthVoiceDone.get(context).get(voiceNum)){
    			fillTree( tempTree, ArrayUtils.subarray(relPitchArray, 0, relPitchArray.length-1));
    			lengthVoiceDone.get(context).set(voiceNum,Boolean.TRUE);
    		}
    		// get short term probability
    		double probST = 0;
    		if( relPitchArray.length > 1 )
    			tempTree.getBackoffProb( tempModel, relPitchArray );
    		if(useSTM == Type.SHORT)
    			return probST;
    		else
    			return (probST + probLT)/2;
		}
	}

	Node<Double> tempTree;
	SimpleLM tempModel;

	static List<List<List<List<Double>>>> trainingPieces = new ArrayList<List<List<List<Double>>>>();
	static{	
		Runtime.getRuntime().addShutdownHook(new Thread(){
			 @Override
			public void run() {
				File f = new File("/Users/tweyde/Documents/workspace2/melody-model/src/data/Melody3v.xml"); 
				ToolBox.storeObject(trainingPieces, f);
	         }
		});
	}
	
	@Override
	public void trainModel(List<List<List<Double>>> melodyList) {
		trainingPieces.add(melodyList);
		seqTree = new Node<Double>();
		for (List<List<Double>> list2 : melodyList) {
			List<Double> relPitchSlice = sliceList(list2,2);		
			fillTree(relPitchSlice.toArray(new Double[relPitchSlice.size()]));
		}
		initSTmodel();
	}

	// helpers for modifying and analysing the tree
	abstract class NodeApplicable{
		abstract void applyToNode(Node<?> n);
	}
	
	protected void traverseDepthFirstPre(Node node, NodeApplicable todo){
		todo.applyToNode(node);
		Collection<Node> col = node.getChildren().values();
		for(Node<?> child: col){
			traverseDepthFirstPre(child, todo); 
		}
	}


	@Override
	public void saveModel(File f) {
		try {
			if(f.getParentFile() != null)
				f.getParentFile().mkdirs();
			FileOutputStream fos = new FileOutputStream(f);
			BufferedOutputStream bos = new BufferedOutputStream(fos);
			ObjectOutputStream oos = new ObjectOutputStream(bos);
			oos.writeObject(seqTree);
			oos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}


	@Override
	public void loadModel(File f) {
		
		try {
			FileInputStream fis = new FileInputStream(f);
			BufferedInputStream bis = new BufferedInputStream(fis);			
			ObjectInputStream ois = new ObjectInputStream(bis);
			seqTree = (Node) ois.readObject();
			ois.close();
		} catch (Exception e) {
			e.printStackTrace();
		} 
		initSTmodel();
	}

	public void initSTmodel() {
		tempModel = new SimpleLM();
		tempTree = tempModel.seqTree;
		lengthVoiceDone = new ArrayList<List<Boolean>>();
	}

	@Override
	public void resetShortTermModel() {
		// TODO Auto-generated method stub
		
	}
	
	public static void main2(String[] args) {
		Double[] seq = new Double[]{1.0,1.0,2.0,3.0,4.0,1.0,3.0,4.0};
		SimpleLM slm = new SimpleLM();
		slm.fillTree(seq);
		for (int i = 1; i <= seq.length; i++) {
			double probs[] = slm.seqTree.getProbs(slm, ArrayUtils.subarray(seq, 0, i));
			for (int j = 0; j < probs.length; j++) {
				System.out.format("%f , ", probs[j] ); 
			}
			System.out.println();
		}
		File f = new File("temp.ser");
		Node<Double> seqTree2 = slm.seqTree;
		slm.saveModel(f);
		slm.loadModel(f);
		assert(ObjectCopy.comparePublicProperties(slm.seqTree, seqTree2));
		System.out.println("CE seq1: "+slm.crossEntropy(seq));
		ArrayUtils.reverse(seq);
		for (int i = 1; i <= seq.length; i++) {
			double probs[] = slm.seqTree.getProbs(slm, ArrayUtils.subarray(seq, 0, i));
			for (int j = 0; j < probs.length; j++) {
				System.out.format("%f , ", probs[j] );				
			}
			System.out.println();
		}
		System.out.println("CE seq-rev: "+slm.crossEntropy(seq));
		Double[] seq2 = new Double[6];//{1.0,1.0,2.0,3.0,1.0,3.0};
		for (int i = 0; i < seq2.length; i++) {
			seq2[i] = new Double((int)(Math.random() * 4));
		}
		System.out.println("CE seq2: "+slm.crossEntropy(seq2));		
	}
	
	public static void main(String[] args) {
		SimpleLM slm = new SimpleLM();
		File f = new File("/Users/tweyde/Documents/workspace2/melody-model/src/data/Melody3v.xml"); 
		List<List<List<List<Double>>>> pieces = ToolBox.getStoredObject(new ArrayList<List<List<List<Double>>>>(), f);
		slm.initSTmodel();
		List<Double> voiceProbs = new ArrayList<Double>(5000);
		for (List<List<List<Double>>> piece : pieces) {
			for (List<List<Double>> voice : piece) {
				List<List<Double>> newVoice = new ArrayList<List<Double>>(voice.size());
				slm.initSTmodel();
				for (List<Double> note : voice) {
					newVoice.add(note);
					voiceProbs.add(slm.modelProbability(newVoice, 0));					
				}
			}
		}
		double entropy = 0;
		for (Double p : voiceProbs) {
			entropy += -Math.log(p);
		}
		entropy/=voiceProbs.size();
		System.out.println("entropy: "+entropy);		
	}



}
