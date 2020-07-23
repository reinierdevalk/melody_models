package models;

import java.util.Arrays;

import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;

/**
 * Utility class that presents the XOR operator as a serial stream of values.
 * This is used to predict the next value in the XOR sequence. This provides a
 * simple stream of numbers that can be predicted.
 * 
 * @author jeff
 * 
 */
public class TemporalXOR {

	/**
	 * 1 xor 0 = 1, 0 xor 0 = 0, 0 xor 1 = 1, 1 xor 1 = 0
	 */
	public static final double[] SEQUENCE = { 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 1.0, 1.0, 1.0, 0.0 };

	private double[][] input;
	private double[][] ideal;

	public MLDataSet generate(final int count) {
		this.input = new double[count][1];
		this.ideal = new double[count][1];

		for (int i = 0; i < this.input.length; i++) {
			this.input[i][0] = TemporalXOR.SEQUENCE[i
					% TemporalXOR.SEQUENCE.length];
			this.ideal[i][0] = TemporalXOR.SEQUENCE[(i + 1)
					% TemporalXOR.SEQUENCE.length];
		}
		return new BasicMLDataSet(this.input, this.ideal);
	}
	
	public MLDataSet generateSin(final int count) {
		this.input = new double[count][25];
		this.ideal = new double[count][25];
		
		final double[][] seq = new double[count+1][25];
		for (int i = 0; i < seq.length; i++) {
			double val = Math.sin(i*.1) * 12;			
			int floor = (int) Math.floor(val);
			int ceil = (int) Math.ceil(val);
			if(floor == ceil)
				seq[i][12+floor] = 1;
			else {
    			seq[i][12+floor] = val-floor;
    			seq[i][12+ceil] = ceil-val;
			}
		}

		for (int i = 0; i < this.input.length; i++) {
			this.input[i]= seq[i];
			this.ideal[i]= seq[i+1];
		}
		return new BasicMLDataSet(this.input, this.ideal);
	}

	public MLDataSet generateSin2(final int count) {
		int width = 25;
		this.ideal = new double[count+2][width];
		this.input = new double[count+2][2*width];
		
		for (int i = 0; i < ideal.length; i++) {
			double val = Math.sin(i*.1) * 12;			
			int floor = (int) Math.floor(val);
			int ceil = (int) Math.ceil(val);
			if(floor == ceil)
				ideal[i][12+floor] = 1;
			else {
    			ideal[i][12+floor] = val-floor;
    			ideal[i][12+ceil] = ceil-val;
			}
			if(i>1){
    			for (int j = 0; j < width; j++) {
    				input[i][j] = ideal[i-2][j];				
    			}
    			for (int j = 0; j < width; j++) {
    				input[i][j+width] = ideal[i-1][j];				
    			}
			}
		}
		this.input = Arrays.copyOfRange(this.input, 2, this.input.length);
		this.ideal = Arrays.copyOfRange(this.ideal, 2, this.ideal.length);

		return new BasicMLDataSet(this.input, this.ideal);
	}

}