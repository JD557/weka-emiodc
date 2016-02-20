package weka.classifiers.trees.oj48;

/**
 * Class for combining the information gain and gain ratio using their product
 *
 * @author João Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public class ProductOptimizationCrit extends OptimizationCrit{

	@Override
	public double combine(double[] values,boolean[] active) {
		double prod = 1;
		int length = 0;
		for (int i=0;i<values.length;++i) {
			if (active[i]) {
				prod*=(1-values[i]);
			}
		}
		return 1-prod;
	}
}
