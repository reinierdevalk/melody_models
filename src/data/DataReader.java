package data;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
//import auxiliary.AuxiliaryTool;

import tools.ToolBox;

public class DataReader implements Serializable {
	
	private static final long serialVersionUID = 1238156541652706452L;
	
	//List<List<List<List<Double>>>> fvt = new ArrayList<List<List<List<Double>>>>();
	public Map<String,List<List<List<Double>>>> fvtMap = new HashMap<String,List<List<List<Double>>>>();
	public Map<String,List<List<List<Double>>>> tvtMap = new HashMap<String,List<List<List<Double>>>>();
	public Map<String,List<List<List<Double>>>> fvfMap = new HashMap<String,List<List<List<Double>>>>();
	public Map<String,List<List<List<Double>>>> tvfMap = new HashMap<String,List<List<List<Double>>>>();
	
	//"data/Four-voice intabulations/"
	public void readDirectory(String dirPath, Map<String,List<List<List<Double>>>> pieceMap){
		File dir = new File(dirPath);
		String filenames[] = dir.list();
//		Map<String,List<List<List<Double>>>> pieceMap = new HashMap<String,List<List<List<Double>>>>();
		for (int i = 0; i < filenames.length; i++) {
			File f = new File(dir,filenames[i]);
//			StringTokenizer st = new StringTokenizer(filenames[i], "()");
//			String pieceName = st.nextToken();
//			String voiceStr = st.nextToken();
			int index = filenames[i].indexOf("(voice ");
			String pieceName = filenames[i].substring(0,index);
			String voiceStr = filenames[i].substring(index+7,index+8);
			int voice = Integer.parseInt(voiceStr);
			if(!pieceMap.containsKey(pieceName))
				pieceMap.put(pieceName, new ArrayList<List<List<Double>>>());
			ToolBox tb = new ToolBox();
			List<List<Double>> features = tb.getStoredObject(new ArrayList<List<Double>>(),f);
			pieceMap.get(pieceName).add(voice, features);
		}
	}
	
	private DataReader(){
		
	};
	
	public static DataReader getInstance(){
		InputStream is = DataReader.class.getResourceAsStream("data.ser");
		ObjectInputStream ois;
		try {
			ois = new ObjectInputStream(is);
			return (DataReader)ois.readObject();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	// this is to regenerate the binary serialized data. 
	public static void main(String[] args) {
		DataReader dr = new DataReader();
		dr.readDirectory("./data/Four-voice intabulations/",dr.fvtMap);
		dr.readDirectory("./data/Three-voice intabulations/",dr.tvtMap);
		dr.readDirectory("./data/Four-voice fugues/",dr.fvfMap);
		dr.readDirectory("./data/Three-voice fugues/",dr.tvfMap);
		File f = new File("./src/data/data.ser");
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(f));
			oos.writeObject(dr);
			oos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
