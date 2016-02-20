package weka.classifiers.trees.oj48;

/**
 * Class for combining the information gain and gain ratio by calculating the
 * euclidean distance to the minimum possible value
 *
 * @author Jo�o Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public class EuclideanMinOptimizationCrit extends OptimizationCrit{

	@Override
	public double combine(double[] values,boolean[] active) {
		double sum = 0;
		for (int i=0;i<values.length;++i) {
			if (active[i]) {
				sum+=values[i]*values[i];
			}
		}
		return Math.sqrt(sum);
	}

}
