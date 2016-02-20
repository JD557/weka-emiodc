package weka.classifiers.trees.oj48;

import java.io.Serializable;

import weka.core.Tag;


/**
 * Class for combining the information gain and gain ratio
 *
 * @author João Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public abstract class OptimizationCrit implements Serializable{

	private static final long serialVersionUID = 6752992755244932479L;
	public static final Tag[] TAGS_RULES = {
		new Tag(0, "SUM", "Sum"),
		new Tag(1, "AVG", "Average"),
		new Tag(2, "PROD", "Product"),
		new Tag(3, "GEOMEAM", "Geometric Mean"),
		new Tag(4, "MAXMIN", "Minimum"),
		new Tag(5, "MAXMAX", "Maximum"),
		new Tag(6, "EUCLMIN", "Euclidean Distance to the Minimum"),
		new Tag(7, "EUCLMAX", "Euclidean Distance to the Maximum")
	};
	
	public abstract double combine(double[] values,boolean[] active);
	public static OptimizationCrit create(int i) {
		switch (i) {
			case 0: return new SumOptimizationCrit();
			case 1: return new AvgOptimizationCrit();
			case 2: return new ProductOptimizationCrit();
			case 3: return new GeometricMeanOptimizationCrit();
			case 4: return new MinOptimizationCrit();
			case 5: return new MaxOptimizationCrit();
			case 6: return new EuclideanMinOptimizationCrit();
			case 7: return new EuclideanMaxOptimizationCrit();
			default: return new SumOptimizationCrit();
		}
	}
}
