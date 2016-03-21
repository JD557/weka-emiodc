Ensemble Methods in Ordinal Classification
==========================================

This repository contains the implementation of the various algorithms presented
in [Ensemble Methods in Ordinal Data Classification](https://repositorio-aberto.up.pt/handle/10216/73795?locale=en)
for Weka 3.7.

Included Algorithms
-------------------

* oAdaBoost: [oAdaBoost: An AdaBoost variant for Ordinal Data Classification](http://joaocosta.eu/Portfolio/docs/pubs/oAdaboost2015.pdf)
* AdaBoost.OR: [Combining ordinal preferences by boosting](https://www.csie.ntu.edu.tw/~htlin/paper/doc/wspl09adaboostor.pdf)
* AdaBoost.M1w: [How to make AdaBoost.M1 work for weak base classifiers by changing only one line of the code](http://www.en-trust.at/eibl/wp-content/uploads/sites/3/2013/08/Eibl02_ECML_AdaBoostM1W.pdf)
* oDT: [Ensemble Methods in Ordinal Data Classification](https://repositorio-aberto.up.pt/bitstream/10216/73795/2/99372.pdf)
* Ordinal Random Forests: [Ensemble Methods in Ordinal Data Classification](https://repositorio-aberto.up.pt/bitstream/10216/73795/2/99372.pdf)

Instalation
-----------

### Download (Recommended)

1. Download the pre-compiled package from https://github.com/JD557/weka-emiodc/releases
2. Open Weka 3.7 and choose `Tools > Package Manager`
3. Click on `File/URL` (under `Unofficial`) and choose the downloaded .zip
4. Restart Weka
5. You should now an `OrdinalEnsembleMethods` package
6. The new classifiers should now be available

### Compile from source

1. Run `sbt package`
2. Compress the compiled jar (`target/scala-2.10/emiodc_2.10-1.0.jar`) alongside the `Description.props` in a zip
2. Open Weka 3.7 and choose `Tools > Package Manager`
3. Click on `File/URL` (under `Unofficial`) and choose the generated .zip
4. Restart Weka
5. You should now an `OrdinalEnsembleMethods` package
6. The new classifiers should now be available
