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
 *    GainRatioSplitCrit.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.oj48;

import java.util.Random;

import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for computing the modified gain ratio for a given distribution.
 *
 * @author João Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public final class ModifiedGainRatioSplitCrit
extends GainRatioSplitCrit{

	/** for serialization */
	private static final long serialVersionUID = -433336694718670930L;

	/**
	 * Computes entropy of distribution before splitting.
	 */
	public double oldEnt(Distribution bags) {

		double returnValue = 0;
		int j;

		for (j=0;j<bags.numClasses();j++) {
			double prob = bags.perClass(j)/bags.total();
			returnValue = returnValue+logFunc(prob);
		}
		return -returnValue*bags.total(); 
	}

	/**
	 * Computes entropy of distribution after splitting.
	 */
	public final double newEnt(Distribution bags) {

		double returnValue = 0;
		int i,j;

		for (i=0;i<bags.numBags();i++){
			double probBag = bags.perBag(i)/bags.total();
			double sum = 0.0;
			for (j=0;j<bags.numClasses();j++) {
				double probClass = bags.perClassPerBag(i,j)/bags.perBag(i);
				sum = sum+logFunc(probClass);
			}
			returnValue = returnValue + probBag*probBag*sum;
		}
		return -returnValue*bags.total();
	}

	/**
	 * Computes entropy after splitting without considering the
	 * class values.
	 */
	public final double splitEnt(Distribution bags) {

		double returnValue = 0;
		int i;

		for (i=0;i<bags.numBags();i++) {
			double prob = bags.perBag(i)/bags.total();
			returnValue = returnValue+logFunc(prob);
		}
		return -returnValue*bags.total();
	}

}
