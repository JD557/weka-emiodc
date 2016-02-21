name := "EMiODC"

version := "1.0"

javacOptions ++= Seq("-source", "1.7", "-target", "1.7")

libraryDependencies ++= Seq(
  "nz.ac.waikato.cms.weka" % "weka-dev" % "3.7.13"
)
