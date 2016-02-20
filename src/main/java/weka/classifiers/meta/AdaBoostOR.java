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
 *    AdaBoostOR.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.meta;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.oj48.DataReplicator;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 <!-- globalinfo-start -->
 * Class for boosting a ordinal class classifier using the Adaboost.OR method. Only ordinal class problems can be tackled. Often dramatically improves performance, but sometimes overfits.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Hsuan-Tien Lin and Ling Li. Combining ordinal preferences by boosting.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{bib:LinLi2009,
 *  title={Combining ordinal preferences by boosting},
 *  author={Lin, Hsuan-Tien and Li, Ling},
 *  booktitle={Proceedings ECML/PKDD 2009 Workshop on Preference Learning},
 *  pages={69--83},
 *  year={2009}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -P &lt;num&gt;
 *  Percentage of weight mass to base training on.
 *  (default 100, reduce to around 90 speed up)</pre>
 * 
 * <pre> -Q
 *  Use resampling for boosting.</pre>
 * 
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 * 
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.DecisionStump)</pre>
 * 
 * <pre> 
 * Options specific to classifier weka.classifiers.trees.DecisionStump:
 * </pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p>
 *
 * @author João Costa (ei09008@fe.up.pt)
 * @version $Revision: 9186 $ 
 */
public class AdaBoostOR 
extends RandomizableIteratedSingleClassifierEnhancer 
implements WeightedInstancesHandler, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = -1178107808933117974L;

	/** Max num iterations tried to find classifier with non-zero error. */ 
	private static int MAX_NUM_RESAMPLING_ITERATIONS = 10;

	/** Array for storing the weights for the votes. */
	protected double [] m_Betas;

	/** The number of successfully generated base classifiers. */
	protected int m_NumIterationsPerformed;

	/** Weight Threshold. The percentage of weight mass used in training */
	protected int m_WeightThreshold = 100;

	/** Use boosting with reweighting? */
	protected boolean m_UseResampling;

	/** The number of classes */
	protected int m_NumClasses;

	/** a ZeroR model in case no model can be built from the data */
	protected Classifier m_ZeroR;

	/**
	 * Constructor.
	 */
	public AdaBoostOR() {

		m_Classifier = new weka.classifiers.trees.OJ48();
	}

	/**
	 * Returns a string describing classifier
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Class for boosting a ordinal class classifier using the Adaboost "
				+ "OR method. Only ordinal class problems can be tackled. Often "
				+ "dramatically improves performance, but sometimes overfits.\n\n"
				+ "For more information, see\n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation 	result;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "Yoav Freund and Robert E. Schapire");
		result.setValue(Field.TITLE, "Experiments with a new boosting algorithm");
		result.setValue(Field.BOOKTITLE, "Thirteenth International Conference on Machine Learning");
		result.setValue(Field.YEAR, "1996");
		result.setValue(Field.PAGES, "148-156");
		result.setValue(Field.PUBLISHER, "Morgan Kaufmann");
		result.setValue(Field.ADDRESS, "San Francisco");

		return result;
	}

	/**
	 * String describing default classifier.
	 * 
	 * @return the default classifier classname
	 */
	protected String defaultClassifierString() {

		return "weka.classifiers.trees.DecisionStump";
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration listOptions() {

		Vector newVector = new Vector();

		newVector.addElement(new Option(
				"\tPercentage of weight mass to base training on.\n"
						+"\t(default 100, reduce to around 90 speed up)",
						"P", 1, "-P <num>"));

		newVector.addElement(new Option(
				"\tUse resampling for boosting.",
				"Q", 0, "-Q"));

		Enumeration enu = super.listOptions();
		while (enu.hasMoreElements()) {
			newVector.addElement(enu.nextElement());
		}

		return newVector.elements();
	}


	/**
	 * Parses a given list of options. <p/>
	 *
   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -P &lt;num&gt;
	 *  Percentage of weight mass to base training on.
	 *  (default 100, reduce to around 90 speed up)</pre>
	 * 
	 * <pre> -Q
	 *  Use resampling for boosting.</pre>
	 * 
	 * <pre> -S &lt;num&gt;
	 *  Random number seed.
	 *  (default 1)</pre>
	 * 
	 * <pre> -I &lt;num&gt;
	 *  Number of iterations.
	 *  (default 10)</pre>
	 * 
	 * <pre> -D
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console</pre>
	 * 
	 * <pre> -W
	 *  Full name of base classifier.
	 *  (default: weka.classifiers.trees.DecisionStump)</pre>
	 * 
	 * <pre> 
	 * Options specific to classifier weka.classifiers.trees.DecisionStump:
	 * </pre>
	 * 
	 * <pre> -D
	 *  If set, classifier is run in debug mode and
	 *  may output additional info to the console</pre>
	 * 
   <!-- options-end -->
	 *
	 * Options after -- are passed to the designated classifier.<p>
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {

		String thresholdString = Utils.getOption('P', options);
		if (thresholdString.length() != 0) {
			setWeightThreshold(Integer.parseInt(thresholdString));
		} else {
			setWeightThreshold(100);
		}

		setUseResampling(Utils.getFlag('Q', options));

		super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		Vector        result;
		String[]      options;
		int           i;

		result = new Vector();

		if (getUseResampling())
			result.add("-Q");

		result.add("-P");
		result.add("" + getWeightThreshold());

		options = super.getOptions();
		for (i = 0; i < options.length; i++)
			result.add(options[i]);

		return (String[]) result.toArray(new String[result.size()]);
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String weightThresholdTipText() {
		return "Weight threshold for weight pruning.";
	}

	/**
	 * Set weight threshold
	 *
	 * @param threshold the percentage of weight mass used for training
	 */
	public void setWeightThreshold(int threshold) {

		m_WeightThreshold = threshold;
	}

	/**
	 * Get the degree of weight thresholding
	 *
	 * @return the percentage of weight mass used for training
	 */
	public int getWeightThreshold() {

		return m_WeightThreshold;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String useResamplingTipText() {
		return "Whether resampling is used instead of reweighting.";
	}

	/**
	 * Set resampling mode
	 *
	 * @param r true if resampling should be done
	 */
	public void setUseResampling(boolean r) {

		m_UseResampling = r;
	}

	/**
	 * Get whether resampling is turned on
	 *
	 * @return true if resampling output is on
	 */
	public boolean getUseResampling() {

		return m_UseResampling;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return      the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		// class
		result.disableAllClasses();
		result.disableAllClassDependencies();
		if (super.getCapabilities().handles(Capability.NOMINAL_CLASS))
			result.enable(Capability.NOMINAL_CLASS);
		if (super.getCapabilities().handles(Capability.BINARY_CLASS))
			result.enable(Capability.BINARY_CLASS);

		return result;
	}

	/**
	 * Boosting method.
	 *
	 * @param data the training data to be used for generating the
	 * boosted classifier.
	 * @throws Exception if the classifier could not be built successfully
	 */

	public void buildClassifier(Instances data) throws Exception {

		super.buildClassifier(data);

		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		// only class? -> build ZeroR model
		if (data.numAttributes() == 1) {
			System.err.println(
					"Cannot build model (only class attribute present in data!), "
							+ "using ZeroR model instead!");
			m_ZeroR = new weka.classifiers.rules.ZeroR();
			m_ZeroR.buildClassifier(data);
			return;
		}
		else {
			m_ZeroR = null;
		}

		m_NumClasses = data.numClasses();
		buildClassifierWithWeights(data);
	}

	/**
	 * Boosting method. Boosts any classifier that can handle weighted
	 * instances.
	 *
	 * @param data the training data to be used for generating the
	 * boosted classifier.
	 * @throws Exception if the classifier could not be built successfully
	 */
	protected void buildClassifierWithWeights(Instances data) 
			throws Exception {

		Instances trainData, training;
		double epsilon;
		int numInstances = data.numInstances();
		CostMatrix[] costMatrices = new CostMatrix[numInstances];
		for (int i=0;i<costMatrices.length;++i) {
			costMatrices[i]=new CostMatrix(data.classAttribute().numValues());
			for (int y=0;y<costMatrices[i].numRows();++y) {
				for (int x=0;x<costMatrices[i].numColumns();++x) {
					costMatrices[i].setElement(y, x, Math.abs(y-x));
				}
			}
		}
		Random randomInstance = new Random(m_Seed);

		// Initialize data
		m_Betas = new double [m_Classifiers.length];
		m_NumIterationsPerformed = 0;

		// Create a copy of the data so that when the weights are diddled
		// with it doesn't mess up the weights for anyone else
		training = new Instances(data, 0, numInstances);

		// Do boostrap iterations
		for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_Classifiers.length; 
				m_NumIterationsPerformed++) {
			if (m_Debug) {
				System.err.println("Training classifier " + (m_NumIterationsPerformed + 1));
			}
			// Select instances to train the classifier on
			trainData = DataReplicator.replicateData(training,0,costMatrices);

			// Build the classifier
			if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable) {
				((Randomizable) m_Classifiers[m_NumIterationsPerformed]).setSeed(randomInstance.nextInt());
			}
			m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainData);

			// Evaluate the classifier
			epsilon = 0.0;
			int wrong = 0;
			double normalization = 0.0;
			for (int i=0;i<training.numInstances();++i) {
				Instance inst = training.get(i);
				int correct = (int)inst.classValue();
				int predict = (int)m_Classifiers[m_NumIterationsPerformed].classifyInstance(inst);
				if (correct!=predict) {wrong++;}
				epsilon += costMatrices[i].getElement(correct, predict);
				normalization += costMatrices[i].getElement(correct, 0)+costMatrices[i].getElement(correct, inst.numClasses()-1);
			}
			epsilon/=normalization;

			// Stop if error too small or error too big and ignore this model
			if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0)) {
				if (m_NumIterationsPerformed == 0) {
					m_NumIterationsPerformed = 1; // If we're the first we have to to use it
				}
				//System.out.println(epsilon);
				break;
			}
			// Determine the weight to assign to this model
			m_Betas[m_NumIterationsPerformed] = Math.log((1 - epsilon) / epsilon);
			
			double delta = Math.exp(m_Betas[m_NumIterationsPerformed]) -1;
			
			for (int i=0;i<training.numInstances();++i) {
				Instance inst = training.get(i);
				int correct = (int)inst.classValue();
				int predict = (int)m_Classifiers[m_NumIterationsPerformed].classifyInstance(inst);
				double predictedCost = costMatrices[i].getElement(correct, predict);
				if (predict>correct) {
					for (int j=correct+1;j<training.numClasses();++j) {
						double oldCost = costMatrices[i].getElement(correct, j);
						if (j>predict) {
							costMatrices[i].setElement(correct, j, oldCost+delta*predictedCost);
							costMatrices[i].setElement(j, correct, oldCost+delta*predictedCost);
						}
						else {
							costMatrices[i].setElement(correct, j, oldCost+delta*oldCost);
							costMatrices[i].setElement(j, correct, oldCost+delta*oldCost);
						}
					}
				}
				else if (predict<correct) {
					for (int j=0;j<correct;++j) {
						double oldCost = costMatrices[i].getElement(correct, j);
						if (j<predict) {
							costMatrices[i].setElement(correct, j, oldCost+delta*predictedCost);
							costMatrices[i].setElement(j, correct, oldCost+delta*predictedCost);
						}
						else {
							costMatrices[i].setElement(correct, j, oldCost+delta*oldCost);
							costMatrices[i].setElement(j, correct, oldCost+delta*oldCost);
						}
					}
				}
			}
			
			if (m_Debug) {
				System.err.println("\terror rate = " + epsilon
						+"  beta = " + m_Betas[m_NumIterationsPerformed]);
			}

		}
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if instance could not be classified
	 * successfully
	 */
	public double [] distributionForInstance(Instance instance) 
			throws Exception {

		// default model?
		if (m_ZeroR != null) {
			return m_ZeroR.distributionForInstance(instance);
		}

		if (m_NumIterationsPerformed == 0) {
			throw new Exception("No model built");
		}
		double [] sums = new double [instance.numClasses()]; 

		if (m_NumIterationsPerformed == 1) {
			return m_Classifiers[0].distributionForInstance(instance);
		} else {
			
			// Normalize betas
			double betaSum = 0.0;
			for (int i = 0; i < m_Betas.length; i++) {
				betaSum += m_Betas[i];
			}
			for (int i = 0; i < m_NumIterationsPerformed; i++) {
				sums[(int)m_Classifiers[i].classifyInstance(instance)] += m_Betas[i]/betaSum;
			}
			double[] probs = new double[sums.length];// Utils.logs2probs(sums);
			double accum=0;
			for (int i=0;i<sums.length;++i) {
				accum += sums[i];
				if (accum<0.5) {
					probs[i]=0;
				}
				else {
					accum=-1;
					probs[i]=1;
				}
			}
			return probs;
		}
	}

	/**
	 * Returns description of the boosted classifier.
	 *
	 * @return description of the boosted classifier as a string
	 */
	public String toString() {

		// only ZeroR model?
		if (m_ZeroR != null) {
			StringBuffer buf = new StringBuffer();
			buf.append(this.getClass().getName().replaceAll(".*\\.", "") + "\n");
			buf.append(this.getClass().getName().replaceAll(".*\\.", "").replaceAll(".", "=") + "\n\n");
			buf.append("Warning: No model could be built, hence ZeroR model is used:\n\n");
			buf.append(m_ZeroR.toString());
			return buf.toString();
		}

		StringBuffer text = new StringBuffer();

		if (m_NumIterationsPerformed == 0) {
			text.append("AdaBoostM1: No model built yet.\n");
		} else if (m_NumIterationsPerformed == 1) {
			text.append("AdaBoostM1: No boosting possible, one classifier used!\n");
			text.append(m_Classifiers[0].toString() + "\n");
		} else {
			text.append("AdaBoostM1: Base classifiers and their weights: \n\n");
			for (int i = 0; i < m_NumIterationsPerformed ; i++) {
				text.append(m_Classifiers[i].toString() + "\n\n");
				text.append("Weight: " + Utils.roundDouble(m_Betas[i], 2) + "\n\n");
			}
			text.append("Number of performed Iterations: " 
					+ m_NumIterationsPerformed + "\n");
		}

		return text.toString();
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return		the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 9186 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv the options
	 */
	public static void main(String [] argv) {
		runClassifier(new AdaBoostOR(), argv);
	}
}
