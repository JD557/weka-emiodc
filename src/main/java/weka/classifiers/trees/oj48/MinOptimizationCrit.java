package weka.classifiers.trees.oj48;

/**
 * Class for combining the information gain and gain ratio by maximizing the
 * minimum value
 *
 * @author Jo�o Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public class MinOptimizationCrit extends OptimizationCrit{

	@Override
	public double combine(double[] values,boolean[] active) {
		double min = Double.MAX_VALUE;
		for (int i=0;i<values.length;++i) {
			if (active[i]) {
				min=(values[i]<min?values[i]:min);
			}
		}
		return (min==Double.MAX_VALUE)?0:min;
	}

}
