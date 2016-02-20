package weka.classifiers.trees.oj48;

/**
 * Class for combining the information gain and gain ratio by calculating the
 * euclidean distance to the maximum possible value
 *
 * @author Jo�o Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public class EuclideanMaxOptimizationCrit extends OptimizationCrit{

	@Override
	public double combine(double[] values,boolean[] active) {
		double sum = 0;
		for (int i=0;i<values.length;++i) {
			if (active[i]) {
				sum+=(1-values[i])*(1-values[i]);
			}
		}
		return Math.sqrt(active.length)-Math.sqrt(sum);
	}

}
