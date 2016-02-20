package weka.classifiers.trees.oj48;

/**
 * Class for combining the information gain and gain ratio using the geometric mean
 *
 * @author Jo�o Costa (ei09008@fe.up.pt)
 * @version $Revision: 1 $
 */
public class GeometricMeanOptimizationCrit extends OptimizationCrit{

	@Override
	public double combine(double[] values,boolean[] active) {
		double prod = 1;
		int length = 0;
		for (int i=0;i<values.length;++i) {
			if (active[i]) {
				prod*=values[i];
				length++;
			}
		}
		return length>0?Math.pow(prod,1.0/length):0;
	}
}
