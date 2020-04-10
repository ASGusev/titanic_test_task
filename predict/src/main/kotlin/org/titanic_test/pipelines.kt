package org.titanic_test

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable


interface Transformer {
    fun transform(features: List<FeatureValue>): List<FeatureValue>
}


interface Predictor {
    fun predictProbability(features: List<FeatureValue>): Float
    fun predictClass(features: List<FeatureValue>): Long
}


@Serializable
@SerialName("TransformerPipeline")
class TransformerPipeline(private val steps: List<Transformer>): Transformer {
    override fun transform(features: List<FeatureValue>): List<FeatureValue> {
        var transformedFeatures = features
        for (step in steps)
            transformedFeatures = step.transform(transformedFeatures)
        return transformedFeatures
    }
}


interface ColumnDataSource {
    val header: List<String>

    fun getData(): Sequence<List<FeatureValue>>

    fun getColumn(colName: String): Sequence<FeatureValue>

    fun getColumns(colNames: List<String>): Sequence<List<FeatureValue>>

    fun getColumnGroups(colNamesGroups: List<List<String>>): Sequence<List<List<FeatureValue>>>
}


@Serializable
class ColumnTransformer(
    private val transformers: List<Transformer>,
    private val colNames: List<List<String>>
) {
    fun transform(dataSource: ColumnDataSource) =
        dataSource.getColumnGroups(colNames)
            .map { featureGroups ->
                (transformers zip featureGroups).map { (transformer, features) ->
                    transformer.transform(features)
                }.flatten()
            }
}


@Serializable
class EndToEndPipeline(
    private val preprocessor: ColumnTransformer,
    private val commonTransformations: Transformer? = null,
    private val predictor: Predictor
) {
    fun predictClass(dataSource: ColumnDataSource): Sequence<Long> {
        var featuresSequence = preprocessor.transform(dataSource)
        if (commonTransformations != null)
            featuresSequence = featuresSequence.map(commonTransformations::transform)
        return featuresSequence.map(predictor::predictClass)
    }
}

