package org.titanic_test

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import com.github.doyaaaaaken.kotlincsv.dsl.csvWriter
import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import java.io.File


private const val INDEX_ID = 0
private const val INDEX_CLASS = 1
private const val INDEX_GENDER = 3
private const val INDEX_AGE = 4
private const val INDEX_SIBLINGS = 5
private const val INDEX_PARENTS_CHILDREN = 6
private const val INDEX_FARE = 8
private const val INDEX_EMBARKED = 10
private val NUMERIC_FEATURE_INDICES = listOf(INDEX_AGE, INDEX_SIBLINGS, INDEX_PARENTS_CHILDREN, INDEX_FARE)
private val CATEGORICAL_FEATURE_INDICES = listOf(INDEX_CLASS, INDEX_GENDER, INDEX_EMBARKED)
private val RESULTS_HEADER = listOf("PassengerId", "Survived")


@Serializable
class TitanicNumericPreprocessor(private val fill: FillerFloat, private val scale: Scaler) {
    fun transform(features: List<String>): List<Float> = scale.transform(fill.transform(features))
}


@Serializable
class TitanicCategoricalPreprocessor(private val fill: FillerString, private val onehot: OneHotEncoder) {
    fun transform(features: List<String>) = onehot.transform(fill.transform(features))
}


@Serializable
class TitanicPreprocessor(private val numeric: TitanicNumericPreprocessor,
                          private val categorical: TitanicCategoricalPreprocessor) {
    fun transform(features: List<String>): List<Float> {
        val numericFeatures = NUMERIC_FEATURE_INDICES.map { features[it] }
        val categoricalFeatures = CATEGORICAL_FEATURE_INDICES.map { features[it] }
        return numeric.transform(numericFeatures) + categorical.transform(categoricalFeatures)
    }
}


@Serializable
class TitanicPipeline(private val preprocessor: TitanicPreprocessor, private val model: LogisticRegressionModel) {
    fun predict(features: List<String>) = model.predict(preprocessor.transform(features))
}


class CLIArguments(parser: ArgParser) {
    val dataPath by parser.storing("--data_path", help="Path to the test data")
        .default("test.csv")
    val pipelinePath by parser.storing("--pipeline_path", help="Path to the saved pipeline")
        .default("titanic_pipeline.json")
    val outputPath by parser.storing("--output_path", help="Path to save the results")
        .default("submission.csv")
}


@UnstableDefault
fun main(args: Array<String>) {
    val arguments = ArgParser(args).parseInto ( ::CLIArguments )

    val json = Json(JsonConfiguration(isLenient=true))
    val pipeline = json.parse(TitanicPipeline.serializer(), File(arguments.pipelinePath).readText())

    csvWriter().open(arguments.outputPath) {
        writeRow(RESULTS_HEADER)
        csvReader().open(arguments.dataPath) {
            readAllAsSequence()
                .drop(1)  //  Skipping header
                .map { listOf(it[INDEX_ID], pipeline.predict(it)) }
                .forEach(::writeRow)
        }
    }
}