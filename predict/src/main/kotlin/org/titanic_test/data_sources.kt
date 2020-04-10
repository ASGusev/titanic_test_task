package org.titanic_test

import org.apache.commons.csv.CSVFormat
import java.io.FileReader

class CSVDataSource(private val filename: String): ColumnDataSource {
    override val header: List<String> = FileReader(filename).use { reader ->
        CSVFormat.DEFAULT
            .withFirstRecordAsHeader()
            .parse(reader)
            .headerNames
    }

    private fun recordsSequence() = CSVFormat.DEFAULT.withFirstRecordAsHeader()
        .parse(FileReader(filename))
        .iterator()
        .asSequence()

    override fun getData() = recordsSequence().map { record -> record.toList().map { StringValue(it) } }

    override fun getColumn(colName: String) = recordsSequence().map { StringValue(it[colName]) }

    override fun getColumns(colNames: List<String>) =
        recordsSequence().map { record -> colNames.map { StringValue(record[it]) } }

    override fun getColumnGroups(colNamesGroups: List<List<String>>) =
        recordsSequence().map { record ->
            colNamesGroups.map { group ->
                group.map { StringValue(record[it]) }
            }
        }
}