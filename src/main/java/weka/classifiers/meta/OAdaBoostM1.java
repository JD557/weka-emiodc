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
 *    AdaBoostM1.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.meta;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.RandomizableIteratedSingleClassifierEnhancer;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.oj48.DataReplicator;
import weka.classifiers.trees.oj48.OptimizationCrit;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 <!-- globalinfo-start -->
 * Class for boosting a binary class classifier using the oAdaboost method. Only ordinal class problems can be tackled. Often dramatically improves performance, but sometimes overfits.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * João Costa and Jaime Cardoso. oAdaBoost: An AdaBoost variant for Ordinal Classification.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Freund1996,
 *    address = {San Francisco},
 *    author = {Yoav Freund and Robert E. Schapire},
 *    booktitle = {Thirteenth International Conference on Machine Learning},
 *    pages = {148-156},
 *    publisher = {Morgan Kaufmann},
 *    title = {Experiments with a new boosting algorithm},
 *    year = {1996}
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
public class OAdaBoostM1 
extends RandomizableIteratedSingleClassifierEnhancer 
implements WeightedInstancesHandler, Sourcable, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = -1178107808933117974L;

	/** Array for storing the weights for the votes. */
	protected double [][] m_RepBetas; // [Replica][Iteration]
	protected int [] m_SelectedAttributes;
	
	protected Classifier[][] m_ReplicatedClassifiers; // [Replica][Iteration]
	
	/** The number of successfully generated base classifiers. */
	protected int m_NumIterationsPerformed;

	/** Use boosting with reweighting? */
	protected boolean m_UseResampling;

	/** The number of classes */
	protected int m_NumClasses;
	
	/** Optimization criteria */
	private int m_optimizationCrit = 0;
	
	/** Keep replicated attributes */
	private boolean m_keepRepAttributes = false;
	
	/** Use Frank and Hall Distribution */
	private boolean m_frankHall = true;

	/** a ZeroR model in case no model can be built from the data */
	protected Classifier m_ZeroR;

	/**
	 * Constructor.
	 */
	public OAdaBoostM1() {

		m_Classifier = new weka.classifiers.trees.DecisionStump();
	}

	/**
	 * Returns a string describing classifier
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Class for boosting a nominal class classifier using the Adaboost "
				+ "M1 method. Only nominal class problems can be tackled. Often "
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
	 * Select only instances with weights that contribute to 
	 * the specified quantile of the weight distribution
	 *
	 * @param data the input instances
	 * @param quantile the specified quantile eg 0.9 to select 
	 * 90% of the weight mass
	 * @return the selected instances
	 */
	protected Instances selectWeightQuantile(Instances data, double quantile) { 

		int numInstances = data.numInstances();
		Instances trainData = new Instances(data, numInstances);
		double [] weights = new double [numInstances];

		double sumOfWeights = 0;
		for(int i = 0; i < numInstances; i++) {
			weights[i] = data.instance(i).weight();
			sumOfWeights += weights[i];
		}
		double weightMassToSelect = sumOfWeights * quantile;
		int [] sortedIndices = Utils.sort(weights);

		// Select the instances
		sumOfWeights = 0;
		for(int i = numInstances - 1; i >= 0; i--) {
			Instance instance = (Instance)data.instance(sortedIndices[i]).copy();
			trainData.add(instance);
			sumOfWeights += weights[sortedIndices[i]];
			if ((sumOfWeights > weightMassToSelect) && 
					(i > 0) && 
					(weights[sortedIndices[i]] != weights[sortedIndices[i - 1]])) {
				break;
			}
		}
		if (m_Debug) {
			System.err.println("Selected " + trainData.numInstances()
					+ " out of " + numInstances);
		}
		return trainData;
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
		
		newVector.addElement(new Option(
				"\tThe optimization criterion\n"
				+ "\t(default: SUM)", "o", 1, "-o " + Tag.toOptionList(OptimizationCrit.TAGS_RULES)));

		newVector.addElement(new Option(
				"\tUse the replicated attributes.",
				"r", 0, "-r"));
		
		newVector.addElement(new Option(
				"\tUse the Frank and Hall distribution.",
				"d", 0, "-d"));
		
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

		setUseResampling(Utils.getFlag('Q', options));
		
		String optimizationCritString = Utils.getOption('o', options);
		if (optimizationCritString.length() != 0) {
			m_optimizationCrit = Integer.parseInt(optimizationCritString);
		} else {
			m_optimizationCrit = 0;
		}
		
		setKeepRepAttributes(Utils.getFlag('r', options));
		
		setFrankHall(Utils.getFlag('d', options));

		super.setOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		Vector<String> result;
		String[]       options;
		int            i;

		result = new Vector<String>();

		if (getUseResampling()) result.add("-Q");
		
		if (m_optimizationCrit!=0) {
			result.add("-o");result.add(Integer.toString(m_optimizationCrit));
		}
		
		if (m_keepRepAttributes) result.add("-r");
		
		if (m_frankHall) result.add("-d");

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
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String optimizationCritTipText() {
		return "The optimization criterion used.";
	}

	/**
	 * Gets the combination rule used
	 * 
	 * @return the combination rule used
	 */
	public SelectedTag getOptimizationCrit() {
		return new SelectedTag(m_optimizationCrit, OptimizationCrit.TAGS_RULES);
	}

	/**
	 * Sets the combination rule to use. Values other than
	 * 
	 * @param newRule the combination rule method to use
	 */
	public void setOptimizationCrit(SelectedTag newRule) {
		if (newRule.getTags() == OptimizationCrit.TAGS_RULES)
			m_optimizationCrit = newRule.getSelectedTag().getID();
	}
	
	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String keepRepAttributesTipText() {
		return "Use replicated attributes during the split?";
	}

	/**
	 * Gets whether replicated attributes are used during the split
	 * 
	 * @return the combination rule used
	 */
	public boolean getKeepRepAttributes() {
		return m_keepRepAttributes;
	}

	/**
	 * Sets if replicated attributes are used during the split
	 * 
	 * @param keep use replicated attributes
	 */
	public void setKeepRepAttributes(boolean keep) {
		m_keepRepAttributes = keep;
	}
	
	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String frankHallTipText() {
		return "Use the Frank and Hall distribution";
	}

	/**
	 * Gets whether the Frank and Hall distribution is used
	 * 
	 * @return whether the Frank and Hall distribution is used
	 */
	public boolean getFrankHall() {
		return m_frankHall;
	}

	/**
	 * Sets if the Frank and Hall distribution is to be used
	 * 
	 * @param fh use the Frank and Hall distribution
	 */
	public void setFrankHall(boolean fh) {
		m_frankHall = fh;
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
		
		m_ReplicatedClassifiers = new Classifier[m_NumClasses-1][];
		for (int i=0;i<m_NumClasses-1;++i) {
			m_ReplicatedClassifiers[i] = AbstractClassifier.makeCopies(m_Classifier, m_NumIterations);
		}
		
		if ((!m_UseResampling) && !(m_Classifier instanceof WeightedInstancesHandler)) {
			m_UseResampling = false;
		}
		buildClassifierWithWeights(data);
	}


	/**
	 * Sets the weights for the next iteration.
	 * 
	 * @param training the training instances
	 * @param reweight the reweighting factor
	 * @throws Exception if something goes wrong
	 */
	protected void setWeights(Instances training, double reweight[]) 
			throws Exception {
		if (m_keepRepAttributes) {
			double oldSumOfWeights = training.sumOfWeights();
			
			Instances projectedInstances = DataReplicator.projectInstances(training, m_SelectedAttributes[m_NumIterationsPerformed]);
			for (int i=0;i<training.size();++i) {
				Instance instance = (Instance) training.get(i);
				Instance projectedInstance = projectedInstances.get(i);
				if (!Utils.eq(m_Classifiers[m_NumIterationsPerformed].classifyInstance(
						projectedInstance), 
						instance.classValue()))
					instance.setWeight(instance.weight() * reweight[0]);
			}
			
			double newSumOfWeights = training.sumOfWeights();
			
			Enumeration enu = training.enumerateInstances();
			while (enu.hasMoreElements()) {
				Instance instance = (Instance) enu.nextElement();
				instance.setWeight(instance.weight() * oldSumOfWeights
						/ newSumOfWeights);
			}
		}
		else {
			Instances[] replicas = DataReplicator.splitReplicas(training);
			double oldSumOfWeights[] = new double[replicas.length];
			double newSumOfWeights[] = new double[replicas.length];
	
			for (int k=0;k<replicas.length;++k) {
				oldSumOfWeights[k] = replicas[k].sumOfWeights();
			}
			Instances projectedInstances = DataReplicator.projectInstances(training, m_SelectedAttributes[m_NumIterationsPerformed]);
			for (int i=0;i<training.size();++i) {
				Instance instance = (Instance) training.get(i);
				int replica = DataReplicator.getInstanceReplica(instance);
				Instance projectedInstance = projectedInstances.get(i);
				if (!Utils.eq(m_ReplicatedClassifiers[replica][m_NumIterationsPerformed].classifyInstance(
						projectedInstance), 
						instance.classValue()))
					instance.setWeight(instance.weight() * reweight[replica]);
			}
	
			// Renormalize weights
			replicas = DataReplicator.splitReplicas(training);
			for (int k=0;k<replicas.length;++k) {
				newSumOfWeights[k] = replicas[k].sumOfWeights();
			}
			Enumeration enu = training.enumerateInstances();
			while (enu.hasMoreElements()) {
				Instance instance = (Instance) enu.nextElement();
				int replica = DataReplicator.getInstanceReplica(instance);
				instance.setWeight(instance.weight() * oldSumOfWeights[replica] 
						/ newSumOfWeights[replica]);
			}
		}
	}
	
	/**
	 * Initialize the classifiers with the random seed.
	 * 
	 * @param randomInstance Random object
	 */
	protected void initializeClassifiers(Random randomInstance) {
		if (m_keepRepAttributes) {
			if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable)
				((Randomizable) m_Classifiers[m_NumIterationsPerformed]).setSeed(randomInstance.nextInt());
		}
			else {
			for (int k=0;k<m_ReplicatedClassifiers.length;++k) {
				if (m_ReplicatedClassifiers[k][m_NumIterationsPerformed] instanceof Randomizable)
					((Randomizable) m_ReplicatedClassifiers[k][m_NumIterationsPerformed]).setSeed(randomInstance.nextInt());
			}
		}
	}
	
	/**
	 * Train a classifier for a replica
	 * 
	 * @param projInst Instances for each replica (already projected along an attribute)
	 * @param replica Replica number
	 * @param rand Random object
	 * @throws Exception if something goes wrong
	 */
	protected void trainClassifierForReplica(Instances projInst[],int replica, Random rand) throws Exception {
		Instances sample;
		if (m_UseResampling) {
			double[] weights = new double[projInst[replica].numInstances()];
			double weightSum = projInst[replica].sumOfWeights();
			for (int i = 0; i < weights.length; i++) {
				weights[i] = projInst[replica].instance(i).weight()/weightSum;
			}
			sample = projInst[replica].resampleWithWeights(rand, weights);
		}
		else {
			sample = projInst[replica];
		}
		m_ReplicatedClassifiers[replica][m_NumIterationsPerformed]
			.buildClassifier(sample);
	}
	
	/**
	 * Train a classifier on the replicated space
	 * 
	 * @param projInst Replicated instances (already projected along an attribute)
	 * @param rand Random object
	 * @throws Exception if something goes wrong
	 */
	protected void trainClassifier(Instances projInst, Random rand) throws Exception {
		Instances sample;
		if (m_UseResampling) {
			double[] weights = new double[projInst.numInstances()];
			double weightSum = projInst.sumOfWeights();
			for (int i = 0; i < weights.length; i++) {
				weights[i] = projInst.instance(i).weight()/weightSum;
			}
			sample = projInst.resampleWithWeights(rand, weights);
		}
		else {
			sample = projInst;
		}
		m_Classifiers[m_NumIterationsPerformed].buildClassifier(sample);
	}
	
	/**
	 * Pick the best attribute to split on
	 * 
	 * @param replicatedInstances Replicated instances
	 * @param optCrit Optimization criterion
	 * @param randomInstance Random object
	 * @return errors for each replica
	 * @throws Exception if something goes wrong
	 */
	protected double[] pickBestAttribute(Instances replicatedInstances,OptimizationCrit optCrit,Random randomInstance) throws Exception {
		int numReplicas = m_keepRepAttributes?1:m_ReplicatedClassifiers.length;
		double bestEpsilonComb = Double.MAX_VALUE;
		double tempEpsilon[] = new double[numReplicas];
		double bestEpsilon[] = new double[numReplicas];
		boolean activeSplits[] = new boolean[numReplicas];
		Evaluation evaluation;
		
		for (int k=0;k<numReplicas;++k) {
			bestEpsilon[k] = Double.MAX_VALUE;
		}
		
		for (int j=0;j<replicatedInstances.classIndex();++j) {
			double epsilonComb = 0;
			if (m_keepRepAttributes) {
				Instances projectedInstances =
						DataReplicator.projectInstances(replicatedInstances, j);
				trainClassifier(projectedInstances,randomInstance);
				evaluation = new Evaluation(projectedInstances);
				evaluation.evaluateModel(
						m_Classifiers[m_NumIterationsPerformed],
						projectedInstances);
				tempEpsilon[0] = evaluation.errorRate();
				epsilonComb = tempEpsilon[0];
				
			}
			else {
				Instances[] projectedInstances = DataReplicator.splitReplicas(
								DataReplicator.projectInstances(replicatedInstances, j)
						);
				for (int k=0;k<numReplicas;++k) {
					trainClassifierForReplica(projectedInstances,k,randomInstance);
					// Evaluate the classifier
					evaluation = new Evaluation(projectedInstances[k]);
					evaluation.evaluateModel(
							m_ReplicatedClassifiers[k][m_NumIterationsPerformed],
							projectedInstances[k]);
					tempEpsilon[k] = evaluation.errorRate();
					activeSplits[k] = 
							m_NumIterationsPerformed==0 ||
							m_RepBetas[k][m_NumIterationsPerformed-1] > 0;
				}
				epsilonComb = optCrit.combine(tempEpsilon, activeSplits);
			}
			if (epsilonComb<bestEpsilonComb) {
				bestEpsilonComb = epsilonComb;
				bestEpsilon = tempEpsilon.clone();
				m_SelectedAttributes[m_NumIterationsPerformed] = j;
			}
		}
		return bestEpsilon;
	}
	
	/**
	 * Update the betas and weights
	 * 
	 * @param replicatedInstances Replicated instances
	 * @param epsilon Errors of each replica
	 * @throws Exception if something goes wrong
	 */
	protected void updateBetasAndWeights(Instances replicatedInstances,double[] epsilon) throws Exception {

		double repReweight[] = new double [epsilon.length];
		if (m_keepRepAttributes) {
			m_RepBetas[0][m_NumIterationsPerformed] = Math.log((1 - epsilon[0]) / epsilon[0]);
			repReweight[0] = (1 - epsilon[0]) / epsilon[0];
		}
		// Determine the weight to assign to this model
		for (int k=0;k<epsilon.length;k++) {
			 // Keep boosting replica?
			boolean validIteration = 
					m_NumIterationsPerformed == 0 ||
					Utils.grOrEq(m_RepBetas[k][m_NumIterationsPerformed-1],0);
			if (validIteration && !Utils.eq(epsilon[k], 0) && !Utils.grOrEq(epsilon[k],0.5)) {
					m_RepBetas[k][m_NumIterationsPerformed] = Math.log((1 - epsilon[k]) / epsilon[k]);
					repReweight[k] = (1 - epsilon[k]) / epsilon[k];
			}
			else {
				m_RepBetas[k][m_NumIterationsPerformed] = -1;
				repReweight[k] = 1;
			}
		}

		// Update instance weights
		setWeights(replicatedInstances, repReweight);
		
	}
	
	/**
	 * Boosting method.
	 *
	 * @param data the training data to be used for generating the
	 * boosted classifier.
	 * @throws Exception if the classifier could not be built successfully
	 */
	protected void buildClassifierWithWeights(Instances data) 
			throws Exception {

		Instances training;
		int numInstances = data.numInstances();
		Random randomInstance = new Random(m_Seed);
		OptimizationCrit optCrit = OptimizationCrit.create(m_optimizationCrit);

		// Initialize data

		// Create a copy of the data so that when the weights are diddled
		// with it doesn't mess up the weights for anyone else
		training = new Instances(data, 0, numInstances);
		
		Instances replicatedInstances = DataReplicator.replicateData(training, 0, null);
		int numReplicas = m_keepRepAttributes?1:DataReplicator.getNumReplicas(replicatedInstances);
		
		m_RepBetas = new double [numReplicas][m_NumIterations];
		m_SelectedAttributes = new int [m_NumIterations];
		m_NumIterationsPerformed = 0;

		// Do boostrap iterations
		for (m_NumIterationsPerformed = 0; m_NumIterationsPerformed < m_NumIterations; 
				m_NumIterationsPerformed++) {
			if (m_Debug) {
				System.err.println("Training classifier " + (m_NumIterationsPerformed + 1));
			}

			// Build the classifier
			initializeClassifiers(randomInstance);
			double epsilon[] = pickBestAttribute(replicatedInstances, optCrit, randomInstance);
			
			// Retrain classifier and evaluate classifier
			// This is needed, as there is no way to deep copy a classifier
			boolean allZero = true;
			boolean allBad = true;
			if (m_keepRepAttributes) {
				Instances projectedInstances =
					DataReplicator.projectInstances(replicatedInstances, m_SelectedAttributes[m_NumIterationsPerformed]);
				trainClassifier(projectedInstances,randomInstance);
				if (!Utils.eq(epsilon[0], 0)) {allZero = false;}
				if (!Utils.grOrEq(epsilon[0], 0.5)) {allBad = false;}
			}
			else {
				for (int k=0;k<numReplicas;++k) {
					Instances[] projectedInstances = DataReplicator.splitReplicas(
						DataReplicator.projectInstances(replicatedInstances, m_SelectedAttributes[m_NumIterationsPerformed])
					);
					trainClassifierForReplica(projectedInstances,k,randomInstance);
					if (!Utils.eq(epsilon[k], 0)) {allZero = false;}
					if (!Utils.grOrEq(epsilon[k], 0.5)) {allBad = false;}
				}
			}
			// Stop if error too small or error too big and ignore this model
			if (allZero || allBad) {
				if (m_NumIterationsPerformed == 0) {
					m_NumIterationsPerformed = 1; // If we're the first we have to to use it
				}
				break;
			}

			updateBetasAndWeights(replicatedInstances,epsilon);
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
		double [] probs = new double [instance.numClasses()-1];
		Instances replicatedInstance = DataReplicator.replicateInstance(instance);
		if (m_NumIterationsPerformed == 1) {
			Instances projectedInstances =
					DataReplicator.projectInstances(replicatedInstance,m_SelectedAttributes[0]);
			if (m_keepRepAttributes) {
				for (int inst=0;inst<projectedInstances.size();++inst) {
					int replica = DataReplicator.getInstanceReplica(projectedInstances.get(inst));
					probs[replica]=m_Classifiers[0].distributionForInstance(projectedInstances.get(inst))[1];
				}
			}
			else {
				Instances[] replicas = DataReplicator.splitReplicas(projectedInstances);
				for (int k=0;k<replicas.length;++k) {
					probs[k]=m_ReplicatedClassifiers[k][0].distributionForInstance(replicas[k].get(0))[1];
				}
			}
		} else {
			double binSums[][] = new double[instance.numClasses()-1][2];
			for (int k=0;k<m_RepBetas.length;++k) {
				for (int i = 0; i < m_NumIterationsPerformed; i++) {
					Instances projectedInstances =
							DataReplicator.projectInstances(replicatedInstance,m_SelectedAttributes[i]);
					if (m_keepRepAttributes) {
						for (int inst=0;inst<projectedInstances.size();++inst) {
							int replica = DataReplicator.getInstanceReplica(projectedInstances.get(inst));
							int label = (int)m_Classifiers[i].classifyInstance(projectedInstances.get(inst));
							binSums[replica][label]+=m_RepBetas[0][i];
						}
					}
					else {
						Instances[] replicas = DataReplicator.splitReplicas(projectedInstances);
						if (m_RepBetas[k][i]>0) { // if the classifier was boosted
							int label = (int)m_ReplicatedClassifiers[k][i].classifyInstance(replicas[k].get(0));
							binSums[k][label]+=m_RepBetas[k][i];
						}
					}
				}
				double binProbs[] = Utils.logs2probs(binSums[k]);
				probs[k] = binProbs[1];
			}
		}
		if (m_frankHall) {
			return DataReplicator.getClassificationFrankHall(probs);
		}
		else {
			int label = DataReplicator.getClassificationLinLi(probs);
			double[] dist = new double [instance.numClasses()];
			dist[label] = 1;
			return dist;
		}
	}

	/**
	 * Returns the boosted model as Java source code.
	 *
	 * @param className the classname of the generated class
	 * @return the tree as Java source code
	 * @throws Exception if something goes wrong
	 */
	public String toSource(String className) throws Exception {

		if (m_NumIterationsPerformed == 0) {
			throw new Exception("No model built yet");
		}
		if (!(m_Classifiers[0] instanceof Sourcable)) {
			throw new Exception("Base learner " + m_Classifier.getClass().getName()
					+ " is not Sourcable");
		}

		StringBuffer text = new StringBuffer("class ");
		text.append(className).append(" {\n\n");

		text.append("  public static double classify(Object[] i) {\n");

		if (m_NumIterationsPerformed == 1) {
			text.append("    return " + className + "_0.classify(i);\n");
		} else {
			text.append("    double [] sums = new double [" + m_NumClasses + "];\n");
			for (int i = 0; i < m_NumIterationsPerformed; i++) {
				text.append("    sums[(int) " + className + '_' + i 
						+ ".classify(i)] += " + m_RepBetas[i] + ";\n");
			}
			text.append("    double maxV = sums[0];\n" +
					"    int maxI = 0;\n"+
					"    for (int j = 1; j < " + m_NumClasses + "; j++) {\n"+
					"      if (sums[j] > maxV) { maxV = sums[j]; maxI = j; }\n"+
					"    }\n    return (double) maxI;\n");
		}
		text.append("  }\n}\n");

		for (int i = 0; i < m_Classifiers.length; i++) {
			text.append(((Sourcable)m_Classifiers[i])
					.toSource(className + '_' + i));
		}
		return text.toString();
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
			for (int k=0;k<m_ReplicatedClassifiers.length;++k) {
				text.append("Replica " + k + " ==========\n\n");
				for (int i = 0; i < m_NumIterationsPerformed ; i++) {
					text.append("Iteration " + i + ":\n\n");
					if (m_RepBetas[k][i]<0) {break;}
					text.append(m_ReplicatedClassifiers[k][i].toString() + "\n\n");
					text.append("Weight: " + Utils.roundDouble(m_RepBetas[k][i], 2) + "\n\n");
				}
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
		runClassifier(new OAdaBoostM1(), argv);
	}
}
