name := "SparkML"

version := "0.1"

scalaVersion := "2.12.11"
val sparkVersion = "2.4.5"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "edu.trinity" %% "swiftvis2core" % "0.1.0-SNAPSHOT",
    "edu.trinity" %% "swiftvis2core_sjs0.6" % "0.1.0-SNAPSHOT",
    "edu.trinity" %% "swiftvis2fx" % "0.1.0-SNAPSHOT",
    "edu.trinity" %% "swiftvis2js_sjs0.6" % "0.1.0-SNAPSHOT",
    "edu.trinity" %% "swiftvis2jvm" % "0.1.0-SNAPSHOT",
    "edu.trinity" %% "swiftvis2swing" % "0.1.0-SNAPSHOT",
    "edu.trinity" %% "swiftvis2spark" % "0.1.0-SNAPSHOT"
)