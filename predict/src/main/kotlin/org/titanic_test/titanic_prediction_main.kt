package org.titanic_test

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import org.apache.commons.csv.CSVFormat
import java.io.File
import java.io.FileWriter


private const val COLUMN_NAME_ID = "PassengerId"
private const val COLUMN_NAME_SURVIVED = "Survived"


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

    val json = Json(JsonConfiguration(isLenient=true), context=PIPELINE_SERIALIZATION_CONTEXT)
    val pipeline = json.parse(EndToEndPipeline.serializer(), File(arguments.pipelinePath).readText())
    val dataSource = CSVDataSource(arguments.dataPath)

    val ids = dataSource.getColumn(COLUMN_NAME_ID)
    val predictions = pipeline.predictClass(dataSource)

    val outWriter = CSVFormat.DEFAULT.print(FileWriter(arguments.outputPath))
    outWriter.use {
        outWriter.printRecord(COLUMN_NAME_ID, COLUMN_NAME_SURVIVED)
        (ids zip predictions).forEach { (id, prediction) -> outWriter.printRecord(id, prediction) }
    }
}