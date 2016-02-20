package weka.classifiers.trees.oj48;

/**
 * Class for combining the information gain and gain ratio using the average
 *
 * @author João Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public class AvgOptimizationCrit extends OptimizationCrit{

	@Override
	public double combine(double[] values,boolean[] active) {
		double sum = 0;
		int length = 0;
		for (int i=0;i<values.length;++i) {
			if (active[i]) {
				sum+=values[i];
				length++;
			}
		}
		return length>0?sum/length:0;
	}
}
