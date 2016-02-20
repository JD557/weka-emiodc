package weka.classifiers.trees.oj48;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.SortedSet;


import weka.classifiers.CostMatrix;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class DataReplicator {

	// s - s value from the data replication method
	// s = 0 => s = K-1
	public static Instances[] frankHall(Instances data,int s,CostMatrix[] cMatrix) {
		
		int K = data.classAttribute().numValues();
		Instances[] replicas = new Instances[K-1];
		
		// Create Replicas
		
		List<String> binaryValues = new ArrayList<String>();
		binaryValues.add("0");
		binaryValues.add("1");
		
		for (int i=0; i<replicas.length; ++i) {
			Instances replica = new Instances(data);
			int oldClassIndex=replica.classIndex();
			replica.insertAttributeAt(
					new Attribute("Binary Label", binaryValues),
					replica.numAttributes()
			);
			replica.setClassIndex(replica.numAttributes()-1);
			for (int j=0;j<data.size(); ++j) {
				Instance instance = replica.get(j);
				double oldClass = instance.value(oldClassIndex);
				// Lin and Li weights
				try {
					if (cMatrix!=null) {
						double weight =
						    (K-1)*Math.abs(
						        cMatrix[j].getElement((int)oldClass,i)-
						        cMatrix[j].getElement((int)oldClass,i+1)
						    );
						instance.setWeight(weight);
					}
				} catch(Exception e) {} // Keep default weight
				
				
				if (oldClass<=i) {
					instance.setClassValue(binaryValues.get(0));
				}
				else {
					instance.setClassValue(binaryValues.get(1));
				}
			}

			if (s>0) { // Clean extra points
				for (int j=0;j<replica.size(); ++j) {
					Instance instance = replica.get(j);
					if (instance.value(oldClassIndex)<i-s ||
						instance.value(oldClassIndex)>i+s) {
						replica.delete(j--);
					}
				}
			}

			replica.deleteAttributeAt(oldClassIndex);
			replicas[i]=replica;
		}
		
		return replicas;
	}

	// s - s value from the data replication method
	// s = 0 => s = K-1
	public static Instances replicateData(Instances data, int s, CostMatrix cMatrix[]) {
		Instances[] replicas = frankHall(data,s,cMatrix);
		Instances results = new Instances(replicas[0]);
		results.clear();
		
		int extraAttributes = replicas.length - 1;
		for (int i=0;i<extraAttributes;++i) {
			results.insertAttributeAt(new Attribute("rep"+i), results.numAttributes());
		}
		
		for (int i=0;i<replicas.length;++i) {
			for (int j=0;j<replicas[i].size();++j) {
				Instance instance = new DenseInstance(replicas[i].get(j));
				for (int k=0;k<extraAttributes;++k) {
					instance.insertAttributeAt(instance.numAttributes());
					instance.setValue(instance.numAttributes()-1, 0);
					if (k==i-1) {
						instance.setValue(instance.numAttributes()-1, 1);
					}
				}
				results.add(instance);
			}
		}
		return results;
	}

	public static Instances replicateInstance(Instance instance) {
		return replicateInstance(instance,0);
	}
	
	public static Instances replicateInstance(Instance instance, int s) {
		ArrayList<Attribute> attributes = Collections.list(instance.enumerateAttributes());
		// add dummy class attribute
		attributes.add(instance.classAttribute());
		Instances initialData = new Instances(
				"init",
				attributes,
				1);
		initialData.add(instance);
		initialData.setClassIndex(instance.classIndex());
		return DataReplicator.replicateData(initialData, s,null);
	}
	
	public static Instances[] splitReplicas(Instances data) {
		int numReplicas = getNumReplicas(data);
		Instances[] replicas = new Instances[numReplicas];
		
		// For every replica
		for (int i=0;i<numReplicas;++i) {
			// Copy every instance
			replicas[i]=new Instances(data);
			// Check every instance
			for (int j=0;j<replicas[i].size();++j) {
				Instance instance = replicas[i].get(j);
				// And remove the wrong ones
				boolean allZero = true;
				for (int k=instance.classIndex()+1;k<instance.numAttributes();++k) {
					if (instance.value(k)==1) {
						allZero = false;
						if (k!=instance.classIndex()+i) {
							replicas[i].remove(j);
							j--;
							break;
						}
					}
				}
				if (allZero && i>0) {
					replicas[i].remove(j);
					j--;
				}
			}
		}
		
		return replicas;
	}
	
	public static Instances projectInstances(Instances data,int attIndex) {
		Instances newData = new Instances(data);
		for (int i=0;i<newData.classIndex();++i) {
			if (i!=attIndex) {
				newData.deleteAttributeAt(i);
				if (i<attIndex) {
					attIndex--;
				}
				i--;
			}
		}
		return newData;
	}
	
	public static Distribution[] getDistributions(Instances[] replicas) throws Exception {
		Distribution[] results = new Distribution[replicas.length];
		for (int i=0;i<replicas.length;++i) {
			results[i] = new Distribution(replicas[i]);
		}
		return results;
	}
	
	public static double[] getClassificationFrankHall(double prob[]) {
		double [] dist = new double[prob.length+1];

		dist[0]             = 1-prob[0];
		dist[dist.length-1] = prob[prob.length-1];
		for (int i=1;i<dist.length-1;++i) {
			dist[i]         = prob[i-1]-prob[i];
			if (dist[i]<0) {
				dist[i]=0;
			}
		}

		return dist;
	}
	
	public static int getClassificationLinLi(double prob[]) {
		int classification=0;
		for (int i=0;i<prob.length;++i) {
			if (prob[i]>=0.5) {
				classification++;
			}
		}
		return classification;
	}
	
	public static int getNumReplicas(Instances data) {
		return data.numAttributes()-data.classIndex();
	}
	
	public static int getNumReplicas(Instance data) {
		return data.numAttributes()-data.classIndex();
	}
	
	public static int getInstanceReplica(Instance instance) {
		for (int i=instance.classIndex()+1;i<instance.numAttributes();++i) {
			if (instance.value(i)==1) {return i-instance.classIndex();}
		}
		return 0;
	}
	
	public static boolean isDataReplicated(Instances data) {
		return data.classAttribute().numValues()<=2;
	}
}
