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
 *    BinC45ModelSelection.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.oj48;

import java.util.Enumeration;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for selecting a C4.5-like binary (!) split for a given dataset.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public class BinC45ModelSelection
extends ModelSelection {

	/** for serialization */
	private static final long serialVersionUID = 179170923545122001L;

	/** Minimum number of instances in interval. */
	private int m_minNoObj;               

	/** Use MDL correction? */
	private boolean m_useMDLcorrection;         

	/** The FULL training dataset. */
	private Instances m_allData; 

	/** The criterion to optimize */
	private OptimizationCrit m_optimizationCrit;
	
	/** The number of attributes to use */
	private int m_numAttributes;
	
	/** Random number generator */
	private Random m_rand;
	
	/**
	 * Initializes the split selection method with the given parameters.
	 *
	 * @param minNoObj minimum number of instances that have to occur in
	 * at least two subsets induced by split
	 * @param allData FULL training dataset (necessary for selection of
	 * split points).  
	 * @param useMDLcorrection whether to use MDL adjustement when
	 * finding splits on numeric attributes
	 * @param optimizationCrit Optimization criterion
	 * @param numAttributes Number of attributes
	 * @param rand Random object
	 */
	public BinC45ModelSelection(int minNoObj,Instances allData,
			boolean useMDLcorrection, OptimizationCrit optimizationCrit,
			int numAttributes,Random rand){
		m_minNoObj = minNoObj;
		m_allData = allData;
		m_useMDLcorrection = useMDLcorrection;
		m_optimizationCrit = optimizationCrit;
		m_numAttributes = numAttributes;
		m_rand = rand;
	}

	/**
	 * Sets reference to training data to null.
	 */
	public void cleanup() {

		m_allData = null;
	}

	/**
	 * Selects C4.5-type split for the given dataset.
	 */
	public final ClassifierSplitModel selectModel(Instances data){

		double minResult;
		double minRandResult;
		BinC45Split [] currentModel;
		BinC45Split bestModel = null;
		BinC45Split bestRandModel = null;
		NoSplit noSplitModel = null;
		double averageInfoGain = 0;
		int validModels = 0;
		boolean multiVal = true;
		Distribution checkDistribution;
		double sumOfWeights;
		int i;

		try{

			// Check if all Instances belong to one class or if not
			// enough Instances to split.
			boolean worthySplit = false;
			Instances[] replicatedData = DataReplicator.splitReplicas(data);
			for (int x=0;x<replicatedData.length;++x) {
				checkDistribution = new Distribution(replicatedData[x]);
				if (Utils.grOrEq(checkDistribution.total(),2*m_minNoObj) &&
					!Utils.eq(checkDistribution.total(),
					checkDistribution.perClass(checkDistribution.maxClass()))
					) {
					worthySplit=true;
					break;
				}
				
			}
			Distribution[] checkDistributions = DataReplicator.getDistributions(replicatedData);
			noSplitModel = new NoSplit(checkDistributions);
			if (!worthySplit) {return noSplitModel;}

			// Check if all attributes are nominal and have a 
			// lot of values.
			Enumeration enu = data.enumerateAttributes();
			while (enu.hasMoreElements()) {
				Attribute attribute = (Attribute) enu.nextElement();
				if ((attribute.isNumeric()) ||
						(Utils.sm((double)attribute.numValues(),
								(0.3*(double)m_allData.numInstances())))){
					multiVal = false;
					break;
				}
			}
			currentModel = new BinC45Split[data.numAttributes()];
			sumOfWeights = data.sumOfWeights();

			// For each attribute.
			for (i = 0; i < data.numAttributes(); i++){

				// Apart from class attribute.
				if (i < (data).classIndex()){

					// Get models for current attribute.
					currentModel[i] = new BinC45Split(i,m_minNoObj,sumOfWeights,m_useMDLcorrection,m_optimizationCrit);
					currentModel[i].buildClassifier(data);

					// Check if useful split for current attribute
					// exists and check for enumerated attributes with 
					// a lot of values.
					if (currentModel[i].checkModel())
						if ((data.attribute(i).isNumeric()) ||
								(multiVal || Utils.sm((double)data.attribute(i).numValues(),
										(0.3*(double)m_allData.numInstances())))){
							averageInfoGain = averageInfoGain+currentModel[i].infoGain();
							validModels++;
						}
				}else {
					currentModel[i] = null;
				}
			}

			// Check if any useful split was found.
			if (validModels == 0) {
				return noSplitModel;
			}
			averageInfoGain = averageInfoGain/(double)validModels;
			
			// Pick random attributes
			minResult = 0;
			minRandResult = 0;
			boolean pickedAttributes[] = new boolean[data.classIndex()];
			int attributeBag[] = new int[data.classIndex()];
			if (m_numAttributes >= data.classIndex()) {
				for (i=0;i<pickedAttributes.length;i++) {
					pickedAttributes[i]=true;
				}
			}
			else {
				for (i=0;i<attributeBag.length;i++) {
					attributeBag[i]=i;
					pickedAttributes[i]=false;
				}
				for (i=attributeBag.length-1;i>=attributeBag.length-m_numAttributes;i--) {
					int pick = m_rand.nextInt(i+1);
					pickedAttributes[attributeBag[pick]]=true;
					if (pick != i) {
						attributeBag[pick] = attributeBag[i]; 
					}
				}
			}

			// Find "best" attribute to split on.
			
			for (i=0;i<data.numAttributes();i++){
				if ((i < data.classIndex()) &&
						(currentModel[i].checkModel()))

					// Use 1E-3 here to get a closer approximation to the original
					// implementation.
					if ((currentModel[i].infoGain() >= (averageInfoGain-1E-3)) &&
							Utils.gr(currentModel[i].gainRatio(),minResult)){ 
						bestModel = currentModel[i];
						minResult = currentModel[i].gainRatio();
						if (pickedAttributes[i]) {
							bestRandModel = currentModel[i];
							minRandResult = currentModel[i].gainRatio();
						}
					}
			}
			
			// If picked attributes are useful, use them
			if (Utils.gr(minRandResult,0)) {
				bestModel = bestRandModel;
				minResult = minRandResult;
			}

			// Check if useful split was found.
			if (Utils.eq(minResult,0))
				return noSplitModel;

			// Add all Instances with unknown values for the corresponding
			// attribute to the distribution for the model, so that
			// the complete distribution is stored with the model.
			for (i=0;i<replicatedData.length;++i) {
				bestModel.distributions()[i].
				addInstWithUnknown(replicatedData[i],bestModel.attIndex());
			}

			// Set the split point analogue to C45 if attribute numeric.
			bestModel.setSplitPoint(m_allData);
			return bestModel;
		}catch(Exception e){
			e.printStackTrace();
		}
		return null;
	}

	/**
	 * Selects C4.5-type split for the given dataset.
	 */
	public final ClassifierSplitModel selectModel(Instances train, Instances test) {

		return selectModel(train);
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
