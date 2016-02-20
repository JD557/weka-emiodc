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
 *    ORandomForest.java
 *    Copyright (C) 2001-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.Bagging;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.PartitionGenerator;

/**
 <!-- globalinfo-start -->
 * Class for constructing a forest of ordinal random trees.<br/>
 * Based on Weka's Random Forest.
 * <br/>
 * For more information see: <br/>
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
 * <pre> -I &lt;number of trees&gt;
 *  Number of trees to build.</pre>
 * 
 * <pre> -K &lt;number of features&gt;
 *  Number of features to consider (&lt;1=int(logM+1)).</pre>
 * 
 * <pre> -S
 *  Seed for random number generator.
 *  (default 1)</pre>
 * 
 * <pre> -depth &lt;num&gt;
 *  The maximum depth of the trees, 0 for unlimited.
 *  (default 0)</pre>
 * 
 * <pre> -print
 *  Print the individual trees in the output</pre>
 * 
 * <pre> -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)</pre>
 * 
 *  <pre> -M
 *  Combine results using the median</pre>
 *  
 *  <pre> -O
 *  Use ordinal tree classifier</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * @author João Costa (ei09008@fe.up.pt)
 * @version $Revision: 9186 $
 */
public class ORandomForest 
  extends AbstractClassifier 
  implements OptionHandler, Randomizable, WeightedInstancesHandler, 
             AdditionalMeasureProducer, TechnicalInformationHandler, 
             PartitionGenerator {

  /** for serialization */
  static final long serialVersionUID = 1116839470751428698L;
  
  /** Number of trees in forest. */
  protected int m_numTrees = 10;

  /** Number of features to consider in random feature selection.
      If less than 1 will use int(logM+1) ) */
  protected int m_numFeatures = 0;

  /** The random seed. */
  protected int m_randomSeed = 1;  

  /** Final number of features that were considered in last build. */
  protected int m_KValue = 0;

  /** The bagger. */
  protected Bagging m_bagger = null;
  
  /** The maximum depth of the trees (0 = unlimited) */
  protected int m_MaxDepth = 0;
  
  /** The number of threads to have executing at any one time */
  protected int m_numExecutionSlots = 1;
  
  /** Print the individual trees in the output */
  protected boolean m_printTrees = false;
  
  protected boolean m_useOJ48 = true;
  protected boolean m_useMedian = true;

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return  
        "Class for constructing a forest of random trees.\n\n"
      + "For more information see: \n\n"
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
    
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Leo Breiman");
    result.setValue(Field.YEAR, "2001");
    result.setValue(Field.TITLE, "Random Forests");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "45");
    result.setValue(Field.NUMBER, "1");
    result.setValue(Field.PAGES, "5-32");
    
    return result;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numTreesTipText() {
    return "The number of trees to be generated.";
  }

  /**
   * Get the value of numTrees.
   *
   * @return Value of numTrees.
   */
  public int getNumTrees() {
    
    return m_numTrees;
  }
  
  /**
   * Set the value of numTrees.
   *
   * @param newNumTrees Value to assign to numTrees.
   */
  public void setNumTrees(int newNumTrees) {
    
    m_numTrees = newNumTrees;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numFeaturesTipText() {
    return "The number of attributes to be used in random selection (see RandomTree).";
  }

  /**
   * Get the number of features used in random selection.
   *
   * @return Value of numFeatures.
   */
  public int getNumFeatures() {
    
    return m_numFeatures;
  }
  
  /**
   * Set the number of features to use in random selection.
   *
   * @param newNumFeatures Value to assign to numFeatures.
   */
  public void setNumFeatures(int newNumFeatures) {
    
    m_numFeatures = newNumFeatures;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String seedTipText() {
    return "The random number seed to be used.";
  }

  /**
   * Set the seed for random number generation.
   *
   * @param seed the seed 
   */
  public void setSeed(int seed) {

    m_randomSeed = seed;
  }
  
  /**
   * Gets the seed for the random number generations
   *
   * @return the seed for the random number generation
   */
  public int getSeed() {

    return m_randomSeed;
  }
  
  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String maxDepthTipText() {
    return "The maximum depth of the trees, 0 for unlimited.";
  }

  /**
   * Get the maximum depth of trh tree, 0 for unlimited.
   *
   * @return 		the maximum depth.
   */
  public int getMaxDepth() {
    return m_MaxDepth;
  }
  
  /**
   * Set the maximum depth of the tree, 0 for unlimited.
   *
   * @param value 	the maximum depth.
   */
  public void setMaxDepth(int value) {
    m_MaxDepth = value;
  }
  
  /**
   * Returns the tip text for this property
   * 
   * @return            tip text for this property suitable for
   *                    displaying in the explorer/experimenter gui
   */
  public String printTreesTipText() {
    return "Print the individual trees in the output";
  }
  
  /**
   * Set whether to print the individual ensemble trees in the output
   * 
   * @param print true if the individual trees are to be printed
   */
  public void setPrintTrees(boolean print) {
    m_printTrees = print;
  }
  
  /**
   * Get whether to print the individual ensemble trees in the output
   * 
   * @return true if the individual trees are to be printed
   */
  public boolean getPrintTrees() {
    return m_printTrees;
  }

  /**
   * Gets the out of bag error that was calculated as the classifier was built.
   *
   * @return the out of bag error
   */
  public double measureOutOfBagError() {
    
    if (m_bagger != null) {
      return m_bagger.measureOutOfBagError();
    } else return Double.NaN;
  }
  
  /**
   * Set the number of execution slots (threads) to use for building the
   * members of the ensemble.
   *
   * @param numSlots the number of slots to use.
   */
  public void setNumExecutionSlots(int numSlots) {
    m_numExecutionSlots = numSlots;
  }

  /**
   * Get the number of execution slots (threads) to use for building
   * the members of the ensemble.
   *
   * @return the number of slots to use
   */
  public int getNumExecutionSlots() {
    return m_numExecutionSlots;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String UseMedianTipText() {
    return "Combine results using the median";
  }
  public void setUseMedian(boolean useMedian) {
    m_useMedian = useMedian;
  }
  public boolean getUseMedian() {
    return m_useMedian;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String UseOJ48TipText() {
    return "Use ordinal tree classifier";
  }
  public void setUseOJ48(boolean useOJ48) {
    m_useOJ48 = useOJ48;
  }
  public boolean getUseOJ48() {
    return m_useOJ48;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numExecutionSlotsTipText() {
    return "The number of execution slots (threads) to use for " +
      "constructing the ensemble.";
  }
  
  /**
   * Returns an enumeration of the additional measure names.
   *
   * @return an enumeration of the measure names
   */
  public Enumeration enumerateMeasures() {
    
    Vector newVector = new Vector(1);
    newVector.addElement("measureOutOfBagError");
    return newVector.elements();
  }
  
  /**
   * Returns the value of the named measure.
   *
   * @param additionalMeasureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  public double getMeasure(String additionalMeasureName) {
    
    if (additionalMeasureName.equalsIgnoreCase("measureOutOfBagError")) {
      return measureOutOfBagError();
    }
    else {throw new IllegalArgumentException(additionalMeasureName 
					     + " not supported (RandomForest)");
    }
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options
   */
  public Enumeration listOptions() {
    
    Vector newVector = new Vector();

    newVector.addElement(new Option(
	"\tNumber of trees to build.",
	"I", 1, "-I <number of trees>"));
    
    newVector.addElement(new Option(
	"\tNumber of features to consider (<1=int(logM+1)).",
	"K", 1, "-K <number of features>"));
    
    newVector.addElement(new Option(
	"\tSeed for random number generator.\n"
	+ "\t(default 1)",
	"S", 1, "-S"));

    newVector.addElement(new Option(
	"\tThe maximum depth of the trees, 0 for unlimited.\n"
	+ "\t(default 0)",
	"depth", 1, "-depth <num>"));
    
    newVector.addElement(new Option(
        "\tPrint the individual trees in the output", "print", 0, "-print"));
    
    newVector.addElement(new Option(
        "\tNumber of execution slots.\n"
        + "\t(default 1 - i.e. no parallelism)",
        "num-slots", 1, "-num-slots <num>"));
    
    newVector.addElement(new Option(
            "\tCombine results using the median",
            "M", 0, "-M"));
    
    newVector.addElement(new Option(
            "\tUse ordinal tree classifier",
            "O", 0, "-O"));


    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }

    return newVector.elements();
  }

  /**
   * Gets the current settings of the forest.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result = new Vector();
    
    result.add("-I");
    result.add("" + getNumTrees());
    
    result.add("-K");
    result.add("" + getNumFeatures());
    
    result.add("-S");
    result.add("" + getSeed());
    
    if (getMaxDepth() > 0) {
      result.add("-depth");
      result.add("" + getMaxDepth());
    }
    
    if (m_printTrees) {
      result.add("-print");
    }
    
    if (getUseMedian()) {
    	result.add("-M");
    }
    
    if (getUseOJ48()) {
    	result.add("-O");
    }
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -I &lt;number of trees&gt;
   *  Number of trees to build.</pre>
   * 
   * <pre> -K &lt;number of features&gt;
   *  Number of features to consider (&lt;1=int(logM+1)).</pre>
   * 
   * <pre> -S
   *  Seed for random number generator.
   *  (default 1)</pre>
   * 
   * <pre> -depth &lt;num&gt;
   *  The maximum depth of the trees, 0 for unlimited.
   *  (default 0)</pre>
   * 
   * <pre> -print
   *  Print the individual trees in the output</pre>
   * 
   * <pre> -num-slots &lt;num&gt;
   *  Number of execution slots.
   *  (default 1 - i.e. no parallelism)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception{
    String	tmpStr;
    
    tmpStr = Utils.getOption('I', options);
    if (tmpStr.length() != 0) {
      m_numTrees = Integer.parseInt(tmpStr);
    } else {
      m_numTrees = 10;
    }
    
    tmpStr = Utils.getOption('K', options);
    if (tmpStr.length() != 0) {
      m_numFeatures = Integer.parseInt(tmpStr);
    } else {
      m_numFeatures = 0;
    }
    
    tmpStr = Utils.getOption('S', options);
    if (tmpStr.length() != 0) {
      setSeed(Integer.parseInt(tmpStr));
    } else {
      setSeed(1);
    }
    
    tmpStr = Utils.getOption("depth", options);
    if (tmpStr.length() != 0) {
      setMaxDepth(Integer.parseInt(tmpStr));
    } else {
      setMaxDepth(0);
    }
    
    setPrintTrees(Utils.getFlag("print", options));

    tmpStr = Utils.getOption("num-slots", options);
    if (tmpStr.length() > 0) {
      setNumExecutionSlots(Integer.parseInt(tmpStr));
    } else {
      setNumExecutionSlots(1);
    }
    
    m_useMedian = Utils.getFlag('M', options);
    m_useOJ48 = Utils.getFlag('O', options);
    
    super.setOptions(options);
    
    Utils.checkForRemainingOptions(options);
  }  

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    return new RandomTree().getCapabilities();
  }

  /**
   * Builds a classifier for a set of instances.
   *
   * @param data the instances to train the classifier with
   * @throws Exception if something goes wrong
   */
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    m_bagger = new Bagging();
    if (m_useOJ48) {
	    OJ48 rTree = new OJ48();
	
	    // set up the random tree options
	    m_KValue = m_numFeatures;
	    if (m_KValue < 1) m_KValue = (int) Utils.log2(data.numAttributes())+1;
	    rTree.setNumAtt(m_KValue);
	    rTree.setMaxDepth(getMaxDepth()==0?-1:getMaxDepth());
	
	    // set up the bagger and build the forest
	    m_bagger.setClassifier(rTree);
    }
    else {
        RandomTree rTree = new RandomTree();

        // set up the random tree options
        m_KValue = m_numFeatures;
        if (m_KValue < 1) m_KValue = (int) Utils.log2(data.numAttributes())+1;
        rTree.setKValue(m_KValue);
        rTree.setMaxDepth(getMaxDepth());

        // set up the bagger and build the forest
        m_bagger.setClassifier(rTree);
    }
    m_bagger.setSeed(m_randomSeed);
    m_bagger.setNumIterations(m_numTrees);
    m_bagger.setCalcOutOfBag(true);
    m_bagger.setNumExecutionSlots(m_numExecutionSlots);
    m_bagger.buildClassifier(data);

  }

  /**
   * Returns the class probability distribution for an instance.
   *
   * @param instance the instance to be classified
   * @return the distribution the forest generates for the instance
   * @throws Exception if computation fails
   */
  public double[] distributionForInstance(Instance instance) throws Exception {
	if (m_useMedian) {
		double[] distribution = m_bagger.distributionForInstance(instance);
		double[] newDistribution = new double[distribution.length];
		double accum = 0;
		for (int i=0;i<distribution.length;++i) {
			accum += distribution[i];
			if (accum>0.5) {newDistribution[i]=1;break;}
		}
		return newDistribution;
	}
	else {
		return m_bagger.distributionForInstance(instance);
	}
  }

  /**
   * Outputs a description of this classifier.
   *
   * @return a string containing a description of the classifier
   */
  public String toString() {

    if (m_bagger == null) { 
      return "Random forest not built yet";
    } else {
      StringBuffer temp = new StringBuffer();
      temp.append("Random forest of " + m_numTrees
          + " trees, each constructed while considering "
          + m_KValue + " random feature" + (m_KValue==1 ? "" : "s") + ".\n"
          + "Out of bag error: "
          + Utils.doubleToString(m_bagger.measureOutOfBagError(), 4) + "\n"
          + (getMaxDepth() > 0 ? ("Max. depth of trees: " + getMaxDepth() + "\n") : (""))
          + "\n");
      if (m_printTrees) {
        temp.append(m_bagger.toString());
      }
      return temp.toString();
    }
  }
  
  /**
   * Builds the classifier to generate a partition.
   */
  public void generatePartition(Instances data) throws Exception {
    
    buildClassifier(data);
  }
  
  /**
   * Computes an array that indicates leaf membership
   */
  public double[] getMembershipValues(Instance inst) throws Exception {

    return m_bagger.getMembershipValues(inst);
  }
  
  /**
   * Returns the number of elements in the partition.
   */
  public int numElements() throws Exception {

    return m_bagger.numElements();
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
   * Main method for this class.
   *
   * @param argv the options
   */
  public static void main(String[] argv) {
    runClassifier(new ORandomForest(), argv);
  }
}

