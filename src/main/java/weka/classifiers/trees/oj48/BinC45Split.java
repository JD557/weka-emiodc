/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    BinC45Split.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.oj48;

import java.util.Enumeration;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class implementing a binary C4.5-like split on an attribute.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class BinC45Split
extends ClassifierSplitModel {

	/** for serialization */
	private static final long serialVersionUID = -1278776919563022474L;

	/** Attribute to split on. */
	private int m_attIndex;        

	/** Minimum number of objects in a split.   */ 
	private int m_minNoObj;         

	/** Use MDL correction? */
	private boolean m_useMDLcorrection;         

	/** Value of split point. */
	private double[] m_splitPoint;

	/** InfoGain of split. */
	private double[] m_infoGain; 

	/** GainRatio of split.  */
	private double[] m_gainRatio;

	/** The sum of the weights of the instances. */
	private double m_sumOfWeights;
	
	/** The criterion to optimize */
	private OptimizationCrit m_optimizationCrit;

	/** Static reference to splitting criterion. */
	private static InfoGainSplitCrit m_infoGainCrit = new InfoGainSplitCrit();

	/** Static reference to splitting criterion. */
	private static GainRatioSplitCrit m_gainRatioCrit = new GainRatioSplitCrit();

	/**
	 * Initializes the split model.
	 */
	public BinC45Split(int attIndex,int minNoObj,double sumOfWeights,
			boolean useMDLcorrection, OptimizationCrit optimizationCrit) {

		// Get index of attribute to split on.
		m_attIndex = attIndex;

		// Set minimum number of objects.
		m_minNoObj = minNoObj;

		// Set sum of weights;
		m_sumOfWeights = sumOfWeights;

		// Whether to use the MDL correction for numeric attributes
		m_useMDLcorrection = useMDLcorrection;
		
		m_optimizationCrit = optimizationCrit;
	}

	/**
	 * Creates a C4.5-type split on the given data.
	 *
	 * @exception Exception if something goes wrong
	 */
	public void buildClassifier(Instances trainInstances)
			throws Exception {

		int numReplicas = DataReplicator.getNumReplicas(trainInstances);
		// Initialize the remaining instance variables.
		m_numSubsets = 0;
		m_splitPoint = new double[numReplicas];
		m_activeSplit = new boolean[numReplicas];
		m_infoGain = new double[numReplicas];
		m_gainRatio = new double[numReplicas];
		m_replicaDistribution = new Distribution[numReplicas];
		for (int i=0;i<numReplicas;++i) {
			m_splitPoint[i] = Double.MAX_VALUE;
			m_infoGain[i] = 0;
			m_gainRatio[i] = 0;
		}

		// Different treatment for enumerated and numeric
		// attributes.
		if (trainInstances.attribute(m_attIndex).isNominal()){
			handleEnumeratedAttribute(trainInstances);
		}else{
			handleNumericAttribute(trainInstances);
		}
	}    

	/**
	 * Returns index of attribute for which split was generated.
	 */
	public final int attIndex(){

		return m_attIndex;
	}

	/**
	 * Returns the split point (numeric attribute only).
	 * 
	 * @return the split point used for a test on a numeric attribute
	 */
	public double[] splitPoint() {
		return m_splitPoint;
	}

	/**
	 * Returns (C4.5-type) gain ratio for the generated split.
	 */
	public final double gainRatio(){
		return m_optimizationCrit.combine(m_gainRatio,m_activeSplit);
	}

	/**
	 * Creates split on enumerated attribute.
	 *
	 * @exception Exception if something goes wrong
	 */
	private void handleEnumeratedAttribute(Instances trainInstances)
			throws Exception {

		Distribution newDistribution,secondDistribution;
		int numAttValues;
		double currIG,currGR,bestGR=0;
		Instance instance;
		int i;

		numAttValues = trainInstances.attribute(m_attIndex).numValues();
		newDistribution = new Distribution(numAttValues,
				trainInstances.numClasses());

		// Only Instances with known values are relevant.
		Enumeration enu = trainInstances.enumerateInstances();
		while (enu.hasMoreElements()) {
			instance = (Instance) enu.nextElement();
			if (!instance.isMissing(m_attIndex))
				newDistribution.add((int)instance.value(m_attIndex),instance);
		}
		m_distribution = newDistribution;

		// For all values
		for (i = 0; i < numAttValues; i++){

			if (Utils.grOrEq(newDistribution.perBag(i),m_minNoObj)){
				secondDistribution = new Distribution(newDistribution,i);

				// Check if minimum number of Instances in the two
				// subsets.
				if (secondDistribution.check(m_minNoObj)){
					m_numSubsets = 2;
					currIG = m_infoGainCrit.splitCritValue(secondDistribution,
							m_sumOfWeights);
					currGR = m_gainRatioCrit.splitCritValue(secondDistribution,
							m_sumOfWeights,
							currIG);
					if ((i == 0) || Utils.gr(currGR,bestGR)){
						bestGR=currGR;
						
						Instances[] replicatedData = DataReplicator.splitReplicas(trainInstances);
						int replicas = replicatedData.length;
						for (int j=0;j<replicas;++j) {
							m_replicaDistribution[j]=new Distribution(numAttValues,
									trainInstances.numClasses());
							Enumeration enu2 = replicatedData[j].enumerateInstances();
							while (enu2.hasMoreElements()) {
								instance = (Instance) enu2.nextElement();
								if (!instance.isMissing(m_attIndex))
									m_replicaDistribution[j].add((int)instance.value(m_attIndex),instance);
							}
							m_replicaDistribution[j]=new Distribution(m_replicaDistribution[j],i);
							m_infoGain[j] = currIG;
							m_gainRatio[j] = currGR;
							m_splitPoint[j] = (double)i;
							m_activeSplit[j] = true;
						}
					}
				}
			}
		}
	}

	/**
	 * Creates split on numeric attribute.
	 *
	 * @exception Exception if something goes wrong
	 */

	private void handleNumericAttribute(Instances trainInstances)
			throws Exception {
		Instances[] replicas = DataReplicator.splitReplicas(trainInstances);
		Distribution[] dists = DataReplicator.getDistributions(replicas);
		
		for (int i=0;i<replicas.length;++i) {
			replicas[i].sort(replicas[i].attribute(m_attIndex));
			
			// Handle Attributes
			handleNumericAttributeSimple(replicas[i],i);
		}


		for (int i=0;i<replicas.length;++i) {
			if (m_splitPoint[i]>=Double.MAX_VALUE &&
				dists[i].total() > 10 &&
				dists[i].prob(0)-dists[i].prob(1)<=0.2 &&
				dists[i].prob(0)-dists[i].prob(1)>=-0.2
			)
			{
				fixXOR(replicas,i);
			}
		}
	}

	/**
	 * Creates split on numeric attribute.
	 *
	 * @exception Exception if something goes wrong
	 */
	private void handleNumericAttributeSimple(Instances trainInstances, int replica)
			throws Exception {

		int firstMiss;
		int next = 1;
		int last = 0;
		int index = 0;
		int splitIndex = -1;
		double currentInfoGain;
		double defaultEnt;
		double minSplit;
		Instance instance;
		int i;
		m_activeSplit[replica] = false;

		// Current attribute is a numeric attribute.
		m_replicaDistribution[replica] = new Distribution(2,trainInstances.numClasses());
		
		// Only Instances with known values are relevant.
		Enumeration enu = trainInstances.enumerateInstances();
		i = 0;
		while (enu.hasMoreElements()) {
			instance = (Instance) enu.nextElement();
			if (instance.isMissing(m_attIndex))
				break;
			m_replicaDistribution[replica].add(1,instance);
			i++;
		}
		firstMiss = i;

		// Compute minimum number of Instances required in each
		// subset.
		minSplit =  0.1*(m_replicaDistribution[replica].total())/
				((double)trainInstances.numClasses());
		if (Utils.smOrEq(minSplit,m_minNoObj)) 
			minSplit = m_minNoObj;
		else
			if (Utils.gr(minSplit,25)) 
				minSplit = 25;

		// Enough Instances with known values?
		if (Utils.sm((double)firstMiss,2*minSplit))
			return;

		// Compute values of criteria for all possible split
		// indices.
		defaultEnt = m_infoGainCrit.oldEnt(m_replicaDistribution[replica]);
		while (next < firstMiss){

			if (trainInstances.instance(next-1).value(m_attIndex)+1e-5 < 
					trainInstances.instance(next).value(m_attIndex)){ 

				// Move class values for all Instances up to next 
				// possible split point.
				m_replicaDistribution[replica].shiftRange(1,0,trainInstances,last,next);

				// Check if enough Instances in each subset and compute
				// values for criteria.
				if (Utils.grOrEq(m_replicaDistribution[replica].perBag(0),minSplit) && 
					Utils.grOrEq(m_replicaDistribution[replica].perBag(1),minSplit) &&
					Utils.gr(m_replicaDistribution[replica].perClass(0),0) &&
					Utils.gr(m_replicaDistribution[replica].perClass(1),0)){
					currentInfoGain = m_infoGainCrit.
							splitCritValue(m_replicaDistribution[replica],m_sumOfWeights,
									defaultEnt);
					if (Utils.gr(currentInfoGain,m_infoGain[replica])){
						m_infoGain[replica] = currentInfoGain;
						splitIndex = next-1;
					}
					index++;
				}
				last = next;
			}
			next++;
		}

		// Was there any useful split?
		if (index == 0) {
			return;
		}

		// Compute modified information gain for best split.
		if (m_useMDLcorrection) {
			m_infoGain[replica] = m_infoGain[replica]-(Utils.log2(index)/m_sumOfWeights);
		}
		if (Utils.smOrEq(m_infoGain[replica],0)) {
			return;
		}

		// Set instance variables' values to values for
		// best split.
		m_activeSplit[replica] = true;
		m_numSubsets = 2;
		m_splitPoint[replica] = 
				(trainInstances.instance(splitIndex+1).value(m_attIndex)+
						trainInstances.instance(splitIndex).value(m_attIndex))/2;

		// In case we have a numerical precision problem we need to choose the
		// smaller value
		if (m_splitPoint[replica] == trainInstances.instance(splitIndex + 1).value(m_attIndex)) {
			m_splitPoint[replica] = trainInstances.instance(splitIndex).value(m_attIndex);
		}

		// Restore distribution for best split.
		m_replicaDistribution[replica] = new Distribution(2,trainInstances.numClasses());
		m_replicaDistribution[replica].addRange(0,trainInstances,0,splitIndex+1);
		m_replicaDistribution[replica].addRange(1,trainInstances,splitIndex+1,firstMiss);

		// Compute modified gain ratio for best split.
		m_gainRatio[replica] = m_gainRatioCrit.
				splitCritValue(m_replicaDistribution[replica],m_sumOfWeights,
						m_infoGain[replica]);
	}

	/**
   * Function to be called if a possible XOR is detected.
   * (See '4.3.1 The XOR Problem' in 'Ensemble Methods for Ordinal Data Classification')
   *
   * This function is currently commented out, as it is hard to tell if this solution
   * is general enough.
   */
	private void fixXOR(Instances[] replicas, int replica)
			throws Exception  {
				/*m_infoGainCrit = new ModifiedInfoGainSplitCrit();
				m_gainRatioCrit = new ModifiedGainRatioSplitCrit();
				if (replicas[replica].attribute(m_attIndex).isNumeric()){
					handleNumericAttributeSimple(replicas[replica],replica);
				}
				m_infoGainCrit = new InfoGainSplitCrit();
				m_gainRatioCrit = new GainRatioSplitCrit();*/
			}

	/**
	 * Returns (C4.5-type) information gain for the generated split.
	 */
	public final double infoGain(){
		return m_optimizationCrit.combine(m_infoGain,m_activeSplit);
	}

	/**
	 * Prints left side of condition.
	 * 
	 * @param data the data to get the attribute name from.
	 * @return the attribute name
	 */
	public final String leftSide(Instances data){

		return data.attribute(m_attIndex).name();
	}

	/**
	 * Prints the condition satisfied by instances in a subset.
	 *
	 * @param index of subset and training set.
	 */
	public final String rightSide(int index,Instances data){
		
		StringBuffer splitPoint = new StringBuffer();

		StringBuffer text;

		text = new StringBuffer();
		if (data.attribute(m_attIndex).isNominal()){
			if (index == 0)
				text.append(" = " + data.attribute(m_attIndex).value((int)m_splitPoint[0]));
			else
				text.append(" != " + data.attribute(m_attIndex).value((int)m_splitPoint[0]));
		}
		else {
			for(int i=0;i<m_splitPoint.length;++i) {
				splitPoint.append(
					(m_splitPoint[i]==Double.MAX_VALUE?
						"INF":
						m_splitPoint[i])
						+" "
					);
			}
			if (index == 0)
				text.append(" <= ["+splitPoint+"]");
			else
				text.append(" > ["+splitPoint+"]");
		}
		return text.toString();
	}


	/**
	 * Sets split point to greatest value in given data smaller or equal to
	 * old split point.
	 * (C4.5 does this for some strange reason).
	 */
	public final void setSplitPoint(Instances allInstances){
		for (int i=0;i<DataReplicator.getNumReplicas(allInstances);++i) {
			double newSplitPoint = -Double.MAX_VALUE;
			double tempValue;
			Instance instance;
	
			if ((!allInstances.attribute(m_attIndex).isNominal()) &&
					(m_numSubsets > 1)){
				Enumeration enu = allInstances.enumerateInstances();
				while (enu.hasMoreElements()) {
					instance = (Instance) enu.nextElement();
					if (!instance.isMissing(m_attIndex)){
						tempValue = instance.value(m_attIndex);
						if (Utils.gr(tempValue,newSplitPoint) && 
								Utils.smOrEq(tempValue,m_splitPoint[i]))
							newSplitPoint = tempValue;
					}
				}
				if (m_splitPoint[i]<Double.MAX_VALUE) { // No split
					m_splitPoint[i] = newSplitPoint;
				}
			}
		}
	}

	/**
	 * Sets distribution associated with model.
	 */
	public void resetDistribution(Instances data) throws Exception {

		Instances insts = new Instances(data, data.numInstances());
		for (int i = 0; i < data.numInstances(); i++) {
			if (whichSubset(data.instance(i)) > -1) {
				insts.add(data.instance(i));
			}
		}
		Distribution newD = new Distribution(insts, this);
		newD.addInstWithUnknown(data, m_attIndex);
		m_distribution = newD;
	}

	/**
	 * Returns weights if instance is assigned to more than one subset.
	 * Returns null if instance is only assigned to one subset.
	 */
	public final double [] weights(Instance instance){

		double [] weights;
		int i;

		int replica = DataReplicator.getInstanceReplica(instance);
		
		if (instance.isMissing(m_attIndex)){
			weights = new double [m_numSubsets];
			for (i=0;i<m_numSubsets;i++)
				weights [i] = m_replicaDistribution[replica].perBag(i)/m_replicaDistribution[replica].total();
			return weights;
		}else{
			return null;
		}
	}

	/**
	 * Returns index of subset instance is assigned to.
	 * Returns -1 if instance is assigned to more than one subset.
	 *
	 * @exception Exception if something goes wrong
	 */

	public final int whichSubset(Instance instance) throws Exception {
		int replica = DataReplicator.getInstanceReplica(instance);
		if (instance.isMissing(m_attIndex))
			return -1;
		else{
			if (instance.attribute(m_attIndex).isNominal()){
				if ((int)m_splitPoint[0] == (int)instance.value(m_attIndex))
					return 0;
				else
					return 1;
			}else
				if (Utils.smOrEq(instance.value(m_attIndex),m_splitPoint[replica]))
					return 0;
				else
					return 1;
		}
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return		the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 8034 $");
	}
}
