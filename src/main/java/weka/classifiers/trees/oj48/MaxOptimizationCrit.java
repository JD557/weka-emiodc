package weka.classifiers.trees.oj48;

/**
 * Class for combining the information gain and gain ratio by maximizing the
 * maximum value
 *
 * @author Jo�o Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public class MaxOptimizationCrit extends OptimizationCrit{

	@Override
	public double combine(double[] values,boolean[] active) {
		double max = 0;
		for (int i=0;i<values.length;++i) {
			if (active[i]) {
				max=(values[i]>max?values[i]:max);
			}
		}
		return max;
	}

}
