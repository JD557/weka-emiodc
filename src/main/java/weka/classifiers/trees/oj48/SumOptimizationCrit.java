package weka.classifiers.trees.oj48;

/**
 * Class for combining the information gain and gain ratio by summing them
 *
 * @author João Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public class SumOptimizationCrit extends OptimizationCrit{

	@Override
	public double combine(double[] values,boolean[] active) {
		double sum = 0;
		for (int i=0;i<values.length;++i) {
			if (active[i]) {
				sum+=values[i];
			}
		}
		return sum/active.length;
	}

}
