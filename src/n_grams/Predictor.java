package n_grams;

import java.util.*;

public class Predictor {
	private final TreeMap<Character, LinkedList<Integer>> tree = new TreeMap<Character, LinkedList<Integer>>();
	private int longestMatch = -1;

	/**
	 * @return predicted next character of the string
	 */
	public char predict(String str) {
		if (str.length() == 0)
			return '0';
		if (str.length() == 1) {
			longestMatch = 0;
			return str.charAt(0);
		}
		if (tree.containsKey(str.charAt(str.length() - 1))) {
			longestMatch++;
			LinkedList<Integer> pred = tree.get(str.charAt(str.length() - 1));
			tree.clear();
			for (int pos : pred) {
				char c = str.charAt(pos + 1);
				LinkedList<Integer> temp;
				if (tree.containsKey(c))
					temp = tree.get(c);
				else
					temp = new LinkedList<Integer>();
				temp.add(pos + 1);
				tree.put(c, temp);
			}
		} 
		else {
			longestMatch = -1;
			int m = 1;
			int i = 0;
			int[] table = createTable(str);
			while (m + i < str.length()) {
				if (str.charAt(str.length() - i - 1) 
						== str.charAt(str.length() - (m + i) - 1))
					i++;
				else {
					insertPrediction(str.charAt(str.length() - m), 
							i, str.length() - m);
					m = m + i - table[i];
					if (table[i] >= 0)
						i = table[i];
				}
			}
			if (i > 0)
				insertPrediction(str.charAt(str.length() - m), i, str.length() - m);
		}
		char prediction = '0';
		int maxCount = 0;
		Iterator<Character> it = tree.keySet().iterator();
		while (it.hasNext()) {
			char key = it.next();
			int count = tree.get(key).size();
			if (count > maxCount) {
				prediction = key;
				maxCount = count;
			}
		}
		return prediction;
	}

	private void insertPrediction(char c, int i, int pos) {
		if (i > longestMatch) {
			tree.clear();
			longestMatch = i;
		}
		if (i < longestMatch)
			return;
		LinkedList<Integer> pred = null;
		if (tree.containsKey(c))
			pred = tree.get(c);
		else
			pred = new LinkedList<Integer>();
		pred.add(pos);
		tree.put(c, pred);
	}

	private int[] createTable(String str) {
		int pos = 2;
		int cnd = 0;
		int[] table = new int[str.length() - 1];
		table[0] = -1;
		while (pos < str.length() - 1) {
			if (str.charAt(str.length() - pos) == str.charAt(str.length() - cnd - 1)) {
				table[pos] = cnd + 1;
				pos++;
				cnd++;
			} else if (cnd > 0)
				cnd = table[cnd];
			else {
				table[pos] = 0;
				pos++;
			}
		}
		return table;
	}
}