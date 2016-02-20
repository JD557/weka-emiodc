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
 *    OJ48.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.oj48.BinC45ModelSelection;
import weka.classifiers.trees.oj48.C45ModelSelection;
import weka.classifiers.trees.oj48.C45PruneableClassifierTree;
import weka.classifiers.trees.oj48.ClassifierTree;
import weka.classifiers.trees.oj48.DataReplicator;
import weka.classifiers.trees.oj48.ModelSelection;
import weka.classifiers.trees.oj48.OptimizationCrit;
import weka.classifiers.trees.oj48.PruneableClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.PartitionGenerator;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Summarizable;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 <!-- globalinfo-start -->
 * Class for generating a pruned or unpruned C4.5 decision tree with the data replication method.
 * Based on Weka's J48.
 * For more information, see<br/>
 * <br/>
 * João Costa, Ricardo Sousa and Jaime Cardoso (2014). Ensemble Methods on Ordinal Data Classification.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;mastersthesis{costa2014ensemble,
 * title={Ensemble methods in ordinal data classification},
 * author={Costa, Jo{\~a}o and Sousa, Ricardo and Cardoso, Jaime S.},
 * year={2014},
 * school={Faculdade de Engenharia da Universidade do Porto}
* }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -U
 *  Use unpruned tree.</pre>
 * 
 * <pre> -O
 *  Do not collapse tree.</pre>
 * 
 * <pre> -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)</pre>
 * 
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)</pre>
 * 
 * <pre> -R
 *  Use reduced error pruning.</pre>
 * 
 * <pre> -N &lt;number of folds&gt;
 *  Set number of folds for reduced error
 *  pruning. One fold is used as pruning set.
 *  (default 3)</pre>
 * 
 * <pre> -S
 *  Don't perform subtree raising.</pre>
 * 
 * <pre> -L
 *  Do not clean up after the tree has been built.</pre>
 * 
 * <pre> -A
 *  Laplace smoothing for predicted probabilities.</pre>
 * 
 * <pre> -J
 *  Do not use MDL correction for info gain on numeric attributes.</pre>
 * 
 * <pre> -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 *
 * <pre> -s &lt;value&gt;
 *  The s value for the data replication method (0=K-1)</pre>
 *  
 * <pre> -o &lt;AVG|PROD|MAJ|MIN|MAX|MED&gt;
 *  The optimization criterion
 *  (default: AVG) </pre>
 * <pre> -K &lt;attributes&gt;
 *  The number of attributes considered for a split.
 *  -1 = All attributes
 *   0 = log2(num_attributes) - 1 
 * </pre>
 * 
 *  <pre> -depth &lt;num&gt;
 *  The maximum depth of the tree, -1 for unlimited.
 *  (default -1)</pre>
 * 
 <!-- options-end -->
 *
 * @author João Costa (ei09008@fe.up.pt)
 * @version $Revision: 9117 $
 */
public class OJ48 
extends AbstractClassifier 
implements OptionHandler, Drawable, Matchable,
WeightedInstancesHandler, Summarizable, AdditionalMeasureProducer, 
TechnicalInformationHandler, PartitionGenerator {

	/** for serialization */
	static final long serialVersionUID = -217733168393644444L;

	/** The decision tree */
	protected ClassifierTree m_root;

	/** Unpruned tree? */
	private boolean m_unpruned = false;

	/** Collapse tree? */
	private boolean m_collapseTree = true;

	/** Confidence level */
	private float m_CF = 0.25f;

	/** Minimum number of instances */
	private int m_minNumObj = 2;

	/** Use MDL correction? */
	private boolean m_useMDLcorrection = false;         

	/** Determines whether probabilities are smoothed using
      Laplace correction when predictions are generated */
	private boolean m_useLaplace = false;

	/** Use reduced error pruning? */
	private boolean m_reducedErrorPruning = false;

	/** Number of folds for reduced error pruning. */
	private int m_numFolds = 3;

	/** Binary splits on nominal attributes? */
	private boolean m_binarySplits = false;

	/** Subtree raising to be performed? */
	private boolean m_subtreeRaising = true;

	/** Cleanup after the tree has been built. */
	private boolean m_noCleanup = false;

	/** Random number seed for reduced-error pruning. */
	private int m_Seed = 1;

	/** S value from the data replication method (default 0=K-1) **/
	private int m_dataRepS = 0;

	/** Optimization criteria **/
	private int m_optimizationCrit = 0;
	
	/** Generate a probability distribution */
	private boolean m_frankHallDistribution = false;
	
	/** The number of attributes considered for a split.
	 *  -1 = All attributes
	 *   0 = log2(num_attributes) - 1 
	 */
	protected int m_numAtt = -1;
	
	/**The maximum depth of the tree, -1 for unlimited. */
	protected int m_maxDepth = -1;

	/**
	 * Returns a string describing classifier
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return  "Class for generating a pruned or unpruned C4.5 decision tree with the data replication method."
				+ " For more information, see\n\n"
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

		result = new TechnicalInformation(Type.BOOK);
		result.setValue(Field.AUTHOR, "Ross Quinlan");
		result.setValue(Field.YEAR, "1993");
		result.setValue(Field.TITLE, "C4.5: Programs for Machine Learning");
		result.setValue(Field.PUBLISHER, "Morgan Kaufmann Publishers");
		result.setValue(Field.ADDRESS, "San Mateo, CA");

		return result;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return      the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities      result;

		try {
			if (!m_reducedErrorPruning)
				result = new C45PruneableClassifierTree(null, !m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup, m_collapseTree).getCapabilities();
			else
				result = new PruneableClassifierTree(null, !m_unpruned, m_numFolds, !m_noCleanup, m_Seed).getCapabilities();
		}
		catch (Exception e) {
			result = new Capabilities(this);
			result.disableAll();
		}

		result.setOwner(this);

		return result;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances the data to train the classifier with
	 * @throws Exception if classifier can't be built successfully
	 */
	public void buildClassifier(Instances instances)
			throws Exception {
		Instances data;
		
	    if (m_numAtt == -1)
	        {m_numAtt = instances.numAttributes();}
	    else if (m_numAtt == 0)
	        {m_numAtt = (int) Utils.log2(instances.numAttributes()) + 1;}
	    else if (m_numAtt > instances.numAttributes()-1)
	        {m_numAtt = instances.numAttributes()-1;}

		
		if (!DataReplicator.isDataReplicated(instances)) {
			data=DataReplicator.replicateData(instances, m_dataRepS,null);
		}
		else {
			data=instances;
		}
		buildReplicatedClassifier(data);
	}
	
	public void buildReplicatedClassifier(Instances data) 
			throws Exception {
		
		ModelSelection modSelection;
		
		Random rand = data.getRandomNumberGenerator(m_Seed);

		OptimizationCrit optCrit = OptimizationCrit.create(m_optimizationCrit);
		if (m_binarySplits) {
			modSelection = new BinC45ModelSelection(m_minNumObj, data, m_useMDLcorrection, optCrit,m_numAtt,rand);
		}
		else {
			modSelection = new C45ModelSelection(m_minNumObj, data, m_useMDLcorrection, optCrit,m_numAtt,rand);
		}
		if (!m_reducedErrorPruning) {
			m_root = new C45PruneableClassifierTree(modSelection, !m_unpruned, m_CF,
					m_subtreeRaising, !m_noCleanup, m_collapseTree);
		}
		else {
			m_root = new PruneableClassifierTree(modSelection, !m_unpruned, m_numFolds,
					!m_noCleanup, m_Seed);
		}
		if (m_binarySplits) {
			m_root.buildClassifier(data,m_maxDepth);
			((BinC45ModelSelection)modSelection).cleanup();
		}
		else {
			m_root.buildClassifier(data,m_maxDepth);
			((C45ModelSelection)modSelection).cleanup();
		}
	}

	/**
	 * Classifies an instance.
	 *
	 * @param instance the instance to classify
	 * @return the classification for the instance
	 * @throws Exception if instance can't be classified successfully
	 */
	public double classifyInstance(Instance instance) throws Exception {
		Instances replicatedData = DataReplicator.replicateInstance(instance);
		double classification = 0.0;
		for (int i=0;i<replicatedData.numInstances();++i) {
			double result = m_root.classifyInstance(replicatedData.get(i));
			classification+=result;
		}
		return classification;
	}

	/** 
	 * Returns class probabilities for an instance.
	 *
	 * @param instance the instance to calculate the class probabilities for
	 * @return the class probabilities
	 * @throws Exception if distribution can't be computed successfully
	 */
	public final double [] distributionForInstance(Instance instance) 
			throws Exception {

		if (m_frankHallDistribution) {
			Instances replicas = DataReplicator.replicateInstance(instance);
			// P(y>Ci)
			double [] prob = m_root.distributionForMulticlassInstance(instance, m_useLaplace);
			// P(y=Ci)
			return DataReplicator.getClassificationFrankHall(prob);
		}
		else {
			double [] doubles = new double[instance.numClasses()];
			double classification = classifyInstance(instance);
			for (int i=0;i<doubles.length;++i) {
				if (i==classification) {
					doubles[i]=1;
				}
				else {doubles[i]=0;}
			}
			return doubles;
		}
	}

	/**
	 *  Returns the type of graph this classifier
	 *  represents.
	 *  @return Drawable.TREE
	 */   
	public int graphType() {
		return Drawable.TREE;
	}

	/**
	 * Returns graph describing the tree.
	 *
	 * @return the graph describing the tree
	 * @throws Exception if graph can't be computed
	 */
	public String graph() throws Exception {

		return m_root.graph();
	}

	/**
	 * Returns tree in prefix order.
	 *
	 * @return the tree in prefix order
	 * @throws Exception if something goes wrong
	 */
	public String prefix() throws Exception {

		return m_root.prefix();
	}


	/**
	 * Returns an enumeration describing the available options.
	 *
	 * Valid options are: <p>
	 *
	 * -U <br>
	 * Use unpruned tree.<p>
	 *
	 * -C confidence <br>
	 * Set confidence threshold for pruning. (Default: 0.25) <p>
	 *
	 * -M number <br>
	 * Set minimum number of instances per leaf. (Default: 2) <p>
	 *
	 * -R <br>
	 * Use reduced error pruning. No subtree raising is performed. <p>
	 *
	 * -N number <br>
	 * Set number of folds for reduced error pruning. One fold is
	 * used as the pruning set. (Default: 3) <p>
	 *
	 * -B <br>
	 * Use binary splits for nominal attributes. <p>
	 *
	 * -S <br>
	 * Don't perform subtree raising. <p>
	 *
	 * -L <br>
	 * Do not clean up after the tree has been built.
	 *
	 * -A <br>
	 * If set, Laplace smoothing is used for predicted probabilites. <p>
	 *
	 * -Q <br>
	 * The seed for reduced-error pruning. <p>
	 *
	 * -D <br>
	 * Calculate probability distribution from the Frank & Hall approach
	 * (the printed tree might be different from the actual tree). <p>
	 * 
	 * -s value <br>
	 * The s value for the data replication method (0=K-1). <p>
	 *
	 * -o &lt;AVG|PROD|MAJ|MIN|MAX|MED&gt; <br>
	 *  The optimization criterion (default: AVG) <p>
	 *
	 * -K <br>
	 * The number of attributes considered for a split.
	 *  (-1 = All attributes; 
	 *   0 = log2(num_attributes) - 1). <p> 
	 * 
	 * -depth depth;
	 *  The maximum depth of the tree, -1 for unlimited.
	 *  (default -1)
	 * 
	 * @return an enumeration of all the available options.
	 */
	public Enumeration listOptions() {

		Vector newVector = new Vector(16);

		newVector.
		addElement(new Option("\tUse unpruned tree.",
				"U", 0, "-U"));
		newVector.
		addElement(new Option("\tDo not collapse tree.",
				"O", 0, "-O"));
		newVector.
		addElement(new Option("\tSet confidence threshold for pruning.\n" +
				"\t(default 0.25)",
				"C", 1, "-C <pruning confidence>"));
		newVector.
		addElement(new Option("\tSet minimum number of instances per leaf.\n" +
				"\t(default 2)",
				"M", 1, "-M <minimum number of instances>"));
		newVector.
		addElement(new Option("\tUse reduced error pruning.",
				"R", 0, "-R"));
		newVector.
		addElement(new Option("\tSet number of folds for reduced error\n" +
				"\tpruning. One fold is used as pruning set.\n" +
				"\t(default 3)",
				"N", 1, "-N <number of folds>"));
		newVector.
		addElement(new Option("\tUse binary splits only.",
				"B", 0, "-B"));
		newVector.
		addElement(new Option("\tDon't perform subtree raising.",
				"S", 0, "-S"));
		newVector.
		addElement(new Option("\tDo not clean up after the tree has been built.",
				"L", 0, "-L"));
		newVector.
		addElement(new Option("\tLaplace smoothing for predicted probabilities.",
				"A", 0, "-A"));
		newVector.
		addElement(new Option("\tDo not use MDL correction for info gain on numeric attributes.",
				"J", 0, "-J"));
		newVector.
		addElement(new Option("\tSeed for random data shuffling (default 1).",
				"Q", 1, "-Q <seed>"));
		newVector.
		addElement(new Option("\tCalculate probability distribution from the Frank & Hall approach (the printed tree might be different from the actual tree).",
				"D", 0, "-D"));
		newVector.
		addElement(new Option("\tThe s value for the data replication method (default 0=K-1).",
				"s", 1, "-s <value>"));
		newVector.
		addElement(new Option("\tThe optimization criterion\n"
				+ "\t(default: SUM)", "o", 1, "-o " + Tag.toOptionList(OptimizationCrit.TAGS_RULES)));
		newVector.
		addElement(new Option("\tThe number of attributes considered for a split (-1 = All attributes; 0 = log2(num_attributes) - 1).",
				"K", 1, "-K <attributes>"));
		newVector.
		addElement(new Option("\tThe maximum depth of the tree, -1 for unlimited (default -1).",
				"depth", 1, "-depth <depth>"));
		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * 
   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -U
	 *  Use unpruned tree.</pre>
	 * 
	 * <pre> -O
	 *  Do not collapse tree.</pre>
	 * 
	 * <pre> -C &lt;pruning confidence&gt;
	 *  Set confidence threshold for pruning.
	 *  (default 0.25)</pre>
	 * 
	 * <pre> -M &lt;minimum number of instances&gt;
	 *  Set minimum number of instances per leaf.
	 *  (default 2)</pre>
	 * 
	 * <pre> -R
	 *  Use reduced error pruning.</pre>
	 * 
	 * <pre> -N &lt;number of folds&gt;
	 *  Set number of folds for reduced error
	 *  pruning. One fold is used as pruning set.
	 *  (default 3)</pre>
	 * 
	 * <pre> -B
	 *  Use binary splits only.</pre>
	 * 
	 * <pre> -S
	 *  Don't perform subtree raising.</pre>
	 * 
	 * <pre> -L
	 *  Do not clean up after the tree has been built.</pre>
	 * 
	 * <pre> -A
	 *  Laplace smoothing for predicted probabilities.</pre>
	 * 
	 * <pre> -J
	 *  Do not use MDL correction for info gain on numeric attributes.</pre>
	 * 
	 * <pre> -Q &lt;seed&gt;
	 *  Seed for random data shuffling (default 1).</pre>
	 *
	 * <pre> -D
	 * Calculate probability distribution from the Frank & Hall approach
	 * (the printed tree might be different from the actual tree). </pre>
	 * 
	 * <pre> -s &lt;value&gt;
	 * The s value for the data replication method (0=K-1) </pre> 
	 *
	 * <pre>
	 * -o &lt;AVG|PROD|MAJ|MIN|MAX|MED&gt;
	 *  The optimization criterion
	 *  (default: AVG) </pre>
	 *  
	 * <pre> -K &lt;attributes&gt;
	 *  The number of attributes considered for a split.
	 *  -1 = All attributes
	 *   0 = log2(num_attributes) - 1 </pre>
	 *
	 * <pre> -depth &lt;num&gt;
	 *  The maximum depth of the tree, -1 for unlimited.
	 *  (default -1)</pre>
	 *
   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {

		// Other options
		String minNumString = Utils.getOption('M', options);
		if (minNumString.length() != 0) {
			m_minNumObj = Integer.parseInt(minNumString);
		} else {
			m_minNumObj = 2;
		}
		m_binarySplits = Utils.getFlag('B', options);
		m_useLaplace = Utils.getFlag('A', options);
		m_useMDLcorrection = !Utils.getFlag('J', options);

		// Pruning options
		m_unpruned = Utils.getFlag('U', options);
		m_collapseTree = !Utils.getFlag('O', options);
		m_subtreeRaising = !Utils.getFlag('S', options);
		m_noCleanup = Utils.getFlag('L', options);
		if ((m_unpruned) && (!m_subtreeRaising)) {
			throw new Exception("Subtree raising doesn't need to be unset for unpruned tree!");
		}
		m_reducedErrorPruning = Utils.getFlag('R', options);
		if ((m_unpruned) && (m_reducedErrorPruning)) {
			throw new Exception("Unpruned tree and reduced error pruning can't be selected " +
					"simultaneously!");
		}
		String confidenceString = Utils.getOption('C', options);
		if (confidenceString.length() != 0) {
			if (m_reducedErrorPruning) {
				throw new Exception("Setting the confidence doesn't make sense " +
						"for reduced error pruning.");
			} else if (m_unpruned) {
				throw new Exception("Doesn't make sense to change confidence for unpruned "
						+"tree!");
			} else {
				m_CF = (new Float(confidenceString)).floatValue();
				if ((m_CF <= 0) || (m_CF >= 1)) {
					throw new Exception("Confidence has to be greater than zero and smaller " +
							"than one!");
				}
			}
		} else {
			m_CF = 0.25f;
		}
		String numFoldsString = Utils.getOption('N', options);
		if (numFoldsString.length() != 0) {
			if (!m_reducedErrorPruning) {
				throw new Exception("Setting the number of folds" +
						" doesn't make sense if" +
						" reduced error pruning is not selected.");
			} else {
				m_numFolds = Integer.parseInt(numFoldsString);
			}
		} else {
			m_numFolds = 3;
		}
		String seedString = Utils.getOption('Q', options);
		if (seedString.length() != 0) {
			m_Seed = Integer.parseInt(seedString);
		} else {
			m_Seed = 1;
		}
		
		m_frankHallDistribution = Utils.getFlag('D', options);

		String dataRepSString = Utils.getOption('s', options);
		if (dataRepSString.length() != 0) {
			m_dataRepS = Integer.parseInt(dataRepSString);
		} else {
			m_dataRepS = 0;
		}

		String optimizationCritString = Utils.getOption('o', options);
		if (optimizationCritString.length() != 0) {
			m_optimizationCrit = Integer.parseInt(optimizationCritString);
		} else {
			m_optimizationCrit = 0;
		}
		
		String numAttString = Utils.getOption('K', options);
		if (numAttString.length() != 0) {
			m_numAtt = Integer.parseInt(numAttString);
		} else {
			m_numAtt = -1;
		}
		
		String maxDepthString = Utils.getOption("depth", options);
		if (numAttString.length() != 0) {
			m_maxDepth = Integer.parseInt(maxDepthString);
		} else {
			m_maxDepth = 0;
		}
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String [] getOptions() {

		String [] options = new String [25];
		int current = 0;

		if (m_noCleanup) {
			options[current++] = "-L";
		}
		if (!m_collapseTree) {
			options[current++] = "-O";
		}
		if (m_unpruned) {
			options[current++] = "-U";
		} else {
			if (!m_subtreeRaising) {
				options[current++] = "-S";
			}
			if (m_reducedErrorPruning) {
				options[current++] = "-R";
				options[current++] = "-N"; options[current++] = "" + m_numFolds;
				options[current++] = "-Q"; options[current++] = "" + m_Seed;
			} else {
				options[current++] = "-C"; options[current++] = "" + m_CF;
			}
		}
		if (m_binarySplits) {
			options[current++] = "-B";
		}
		options[current++] = "-M"; options[current++] = "" + m_minNumObj;
		if (m_useLaplace) {
			options[current++] = "-A";
		}
		if (!m_useMDLcorrection) {
			options[current++] = "-J";
		}
		
		if (m_frankHallDistribution) {
			options[current++] = "-D";
		}

		if (m_dataRepS!=0) {
			options[current++] = "-s"; options[current++] = "" + m_dataRepS;
		}

		if (m_optimizationCrit!=0) {
			options[current++] = "-o"; options[current++] = "" + m_optimizationCrit;
		}
		
		if (m_numAtt!=-1) {
			options[current++] = "-K"; options[current++] = "" + m_numAtt;
		}
		
		if (m_maxDepth!=-1) {
			options[current++] = "-depth"; options[current++] = "" + m_maxDepth;
		}

		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String seedTipText() {
		return "The seed used for randomizing the data " +
				"when reduced-error pruning is used.";
	}

	/**
	 * Get the value of Seed.
	 *
	 * @return Value of Seed.
	 */
	public int getSeed() {

		return m_Seed;
	}

	/**
	 * Set the value of Seed.
	 *
	 * @param newSeed Value to assign to Seed.
	 */
	public void setSeed(int newSeed) {

		m_Seed = newSeed;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String useLaplaceTipText() {
		return "Whether counts at leaves are smoothed based on Laplace.";
	}

	/**
	 * Get the value of useLaplace.
	 *
	 * @return Value of useLaplace.
	 */
	public boolean getUseLaplace() {

		return m_useLaplace;
	}

	/**
	 * Set the value of useLaplace.
	 *
	 * @param newuseLaplace Value to assign to useLaplace.
	 */
	public void setUseLaplace(boolean newuseLaplace) {

		m_useLaplace = newuseLaplace;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String useMDLcorrectionTipText() {
		return "Whether MDL correction is used when finding splits on numeric attributes.";
	}

	/**
	 * Get the value of useMDLcorrection.
	 *
	 * @return Value of useMDLcorrection.
	 */
	public boolean getUseMDLcorrection() {

		return m_useMDLcorrection;
	}

	/**
	 * Set the value of useMDLcorrection.
	 *
	 * @param newuseMDLcorrection Value to assign to useMDLcorrection.
	 */
	public void setUseMDLcorrection(boolean newuseMDLcorrection) {

		m_useMDLcorrection = newuseMDLcorrection;
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier
	 */
	public String toString() {

		if (m_root == null) {
			return "No classifier built";
		}
		if (m_unpruned)
			return "oJ48 unpruned tree\n------------------\n" + m_root.toString();
		else
			return "oJ48 pruned tree\n------------------\n" + m_root.toString();
	}

	/**
	 * Returns a superconcise version of the model
	 * 
	 * @return a summary of the model
	 */
	public String toSummaryString() {

		return "Number of leaves: " + m_root.numLeaves() + "\n"
				+ "Size of the tree: " + m_root.numNodes() + "\n";
	}

	/**
	 * Returns the size of the tree
	 * @return the size of the tree
	 */
	public double measureTreeSize() {
		return m_root.numNodes();
	}

	/**
	 * Returns the number of leaves
	 * @return the number of leaves
	 */
	public double measureNumLeaves() {
		return m_root.numLeaves();
	}

	/**
	 * Returns the number of rules (same as number of leaves)
	 * @return the number of rules
	 */
	public double measureNumRules() {
		return m_root.numLeaves();
	}

	/**
	 * Returns an enumeration of the additional measure names
	 * @return an enumeration of the measure names
	 */
	public Enumeration enumerateMeasures() {
		Vector newVector = new Vector(3);
		newVector.addElement("measureTreeSize");
		newVector.addElement("measureNumLeaves");
		newVector.addElement("measureNumRules");
		return newVector.elements();
	}

	/**
	 * Returns the value of the named measure
	 * @param additionalMeasureName the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	public double getMeasure(String additionalMeasureName) {
		if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
			return measureNumRules();
		} else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
			return measureTreeSize();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
			return measureNumLeaves();
		} else {
			throw new IllegalArgumentException(additionalMeasureName 
					+ " not supported (j48)");
		}
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String unprunedTipText() {
		return "Whether pruning is performed.";
	}

	/**
	 * Get the value of unpruned.
	 *
	 * @return Value of unpruned.
	 */
	public boolean getUnpruned() {

		return m_unpruned;
	}

	/**
	 * Set the value of unpruned. Turns reduced-error pruning
	 * off if set.
	 * @param v  Value to assign to unpruned.
	 */
	public void setUnpruned(boolean v) {

		if (v) {
			m_reducedErrorPruning = false;
		}
		m_unpruned = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String collapseTreeTipText() {
		return "Whether parts are removed that do not reduce training error.";
	}

	/**
	 * Get the value of collapseTree.
	 *
	 * @return Value of collapseTree.
	 */
	public boolean getCollapseTree() {

		return m_collapseTree;
	}

	/**
	 * Set the value of collapseTree.
	 * @param v  Value to assign to collapseTree.
	 */
	public void setCollapseTree(boolean v) {

		m_collapseTree = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String confidenceFactorTipText() {
		return "The confidence factor used for pruning (smaller values incur "
				+ "more pruning).";
	}

	/**
	 * Get the value of CF.
	 *
	 * @return Value of CF.
	 */
	public float getConfidenceFactor() {

		return m_CF;
	}

	/**
	 * Set the value of CF.
	 *
	 * @param v  Value to assign to CF.
	 */
	public void setConfidenceFactor(float v) {

		m_CF = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String minNumObjTipText() {
		return "The minimum number of instances per leaf.";
	}

	/**
	 * Get the value of minNumObj.
	 *
	 * @return Value of minNumObj.
	 */
	public int getMinNumObj() {

		return m_minNumObj;
	}

	/**
	 * Set the value of minNumObj.
	 *
	 * @param v  Value to assign to minNumObj.
	 */
	public void setMinNumObj(int v) {

		m_minNumObj = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String reducedErrorPruningTipText() {
		return "Whether reduced-error pruning is used instead of C.4.5 pruning.";
	}

	/**
	 * Get the value of reducedErrorPruning. 
	 *
	 * @return Value of reducedErrorPruning.
	 */
	public boolean getReducedErrorPruning() {

		return m_reducedErrorPruning;
	}

	/**
	 * Set the value of reducedErrorPruning. Turns
	 * unpruned trees off if set.
	 *
	 * @param v  Value to assign to reducedErrorPruning.
	 */
	public void setReducedErrorPruning(boolean v) {

		if (v) {
			m_unpruned = false;
		}
		m_reducedErrorPruning = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String numFoldsTipText() {
		return "Determines the amount of data used for reduced-error pruning. "
				+ " One fold is used for pruning, the rest for growing the tree.";
	}

	/**
	 * Get the value of numFolds.
	 *
	 * @return Value of numFolds.
	 */
	public int getNumFolds() {

		return m_numFolds;
	}

	/**
	 * Set the value of numFolds.
	 *
	 * @param v  Value to assign to numFolds.
	 */
	public void setNumFolds(int v) {

		m_numFolds = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String binarySplitsTipText() {
		return "Whether to use binary splits on nominal attributes when "
				+ "building the trees.";
	}

	/**
	 * Get the value of binarySplits.
	 *
	 * @return Value of binarySplits.
	 */
	public boolean getBinarySplits() {

		return m_binarySplits;
	}

	/**
	 * Set the value of binarySplits.
	 *
	 * @param v  Value to assign to binarySplits.
	 */
	public void setBinarySplits(boolean v) {

		m_binarySplits = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String subtreeRaisingTipText() {
		return "Whether to consider the subtree raising operation when pruning.";
	}

	/**
	 * Get the value of subtreeRaising.
	 *
	 * @return Value of subtreeRaising.
	 */
	public boolean getSubtreeRaising() {

		return m_subtreeRaising;
	}

	/**
	 * Set the value of subtreeRaising.
	 *
	 * @param v  Value to assign to subtreeRaising.
	 */
	public void setSubtreeRaising(boolean v) {

		m_subtreeRaising = v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String saveInstanceDataTipText() {
		return "Whether to save the training data for visualization.";
	}

	/**
	 * Check whether instance data is to be saved.
	 *
	 * @return true if instance data is saved
	 */
	public boolean getSaveInstanceData() {

		return m_noCleanup;
	}

	/**
	 * Set whether instance data is to be saved.
	 * @param v true if instance data is to be saved
	 */
	public void setSaveInstanceData(boolean v) {

		m_noCleanup = v;
	}
	
	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String frankHallDistributionValueTipText() {
		return "Calculate probability distribution from the Frank & Hall approach (the printed tree might be different from the actual tree)";
	}

	/**
	 * Check whether a distribution should be calculated.
	 *
	 * @return true if a distribution should be calculated.
	 */
	public boolean getFrankHallDistribution() {
		return m_frankHallDistribution;
	}

	/**
	 * Set whether a distribution should be calculated.
	 *
	 * @param f  f true if distribution should be calculated.
	 */
	public void setFrankHallDistribution(boolean f) {

		m_frankHallDistribution = f;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String dataRepSValueTipText() {
		return "The s value for the data replication method (default 0=K-1)";
	}

	/**
	 * Get the value of s.
	 *
	 * @return Value of s.
	 */
	public int getDataRepSValue() {
		return m_dataRepS;
	}

	/**
	 * Set the value of s.
	 *
	 * @param s  Value to assign to s.
	 */
	public void setDataRepSValue(int s) {

		m_dataRepS = s;
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
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String numAttTipText() {
		return "The number of attributes considered for a split (-1 = All attributes; 0 = log2(num_attributes) - 1)";
	}

	/**
	 * Get the number of attributes to split on.
	 *
	 * @return Value of s.
	 */
	public int getNumAtt() {
		return m_numAtt;
	}

	/**
	 * Set the number of attributes to split on.
	 *
	 * @param s  Value to assign to s.
	 */
	public void setNumAtt(int k) {

		m_numAtt = k;
	}
	
	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String maxDepthTipText() {
		return "The maximum depth of the tree, -1 for unlimited.";
	}

	/**
	 * Get the maximum depth of the tree.
	 *
	 * @return Maximum depth.
	 */
	public int getMaxDepth() {
		return m_maxDepth;
	}

	/**
	 * Set the maximum depth of the tree.
	 *
	 * @param s  Value to assign to s.
	 */
	public void setMaxDepth(int d) {

		m_maxDepth = d;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return		the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 9117 $");
	}

	/**
	 * Builds the classifier to generate a partition.
	 */
	public void generatePartition(Instances data) throws Exception {

		buildClassifier(data);
	}

	/**
	 * Computes an array that indicates node membership.
	 */
	public double[] getMembershipValues(Instance inst) throws Exception {

		return m_root.getMembershipValues(inst);
	}

	/**
	 * Returns the number of elements in the partition.
	 */
	public int numElements() throws Exception {

		return m_root.numNodes();
	}

	/**
	 * Main method for testing this class
	 *
	 * @param argv the commandline options
	 */
	public static void main(String [] argv){
		runClassifier(new OJ48(), argv);
	}
}

