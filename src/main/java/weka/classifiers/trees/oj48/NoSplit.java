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
 *    NoSplit.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.oj48;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;

/**
 * Class implementing a "no-split"-split.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 8034 $
 */
public final class NoSplit
extends ClassifierSplitModel{

	/** for serialization */
	private static final long serialVersionUID = -1292620749331337546L;

	public NoSplit(Distribution[] distributions){
		m_replicaDistribution = new Distribution[distributions.length];
		for (int i=0;i<distributions.length;++i) {
			m_replicaDistribution[i] = new Distribution(distributions[i]);
		}
		m_numSubsets = 1;
	}

	public double classProb(int classIndex, Instance instance, int theSubset) 
			throws Exception {
		int replica = DataReplicator.getInstanceReplica(instance);
		if (theSubset > -1) {
			return m_replicaDistribution[replica].prob(classIndex,theSubset);
		} else {
			double [] weights = weights(instance);
			if (weights == null) {
				return m_replicaDistribution[replica].prob(classIndex);
			} else {
				double prob = 0;
				for (int i = 0; i < weights.length; i++) {
					prob += weights[i] * m_replicaDistribution[replica].prob(classIndex, i);
				}
				return prob;
			}
		}
	}

	/**
	 * Creates a "no-split"-split for a given set of instances.
	 *
	 * @exception Exception if split can't be built successfully
	 */
	public final void buildClassifier(Instances instances) 
			throws Exception {
		m_distribution = new Distribution(instances);
		m_numSubsets = 1;
	}

	/**
	 * Always returns 0 because only there is only one subset.
	 */
	public final int whichSubset(Instance instance){

		return 0;
	}

	/**
	 * Always returns null because there is only one subset.
	 */
	public final double [] weights(Instance instance){

		return null;
	}

	/**
	 * Does nothing because no condition has to be satisfied.
	 */
	public final String leftSide(Instances instances){

		return "";
	}

	/**
	 * Does nothing because no condition has to be satisfied.
	 */
	public final String rightSide(int index, Instances instances){

		return "";
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
