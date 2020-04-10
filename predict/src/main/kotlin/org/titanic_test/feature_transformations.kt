package org.titanic_test

import kotlinx.serialization.*


private val ONE_HOT_TRUE = FloatValue(1F)
private val ONE_HOT_FALSE = FloatValue(0F)


@Serializable
@SerialName("SimpleImputer")
class Imputer(private val values: List<FeatureValue>): Transformer {
    override fun transform(features: List<FeatureValue>): List<FeatureValue> = (features zip values)
        .map { (feature, fillingValue) -> if (feature.isEmpty) fillingValue else feature }
}


@Serializable
@SerialName("Scale")
class Scaler(private val min: List<Float>, private val range: List<Float>): Transformer {
    override fun transform(features: List<FeatureValue>): List<FeatureValue> {
        val scaledFeatures = mutableListOf<FeatureValue>()
        for (i in 0..features.lastIndex)
            scaledFeatures.add(FloatValue((features[i].floatValue - min[i]) / range[i]))
        return scaledFeatures
    }
}


@Serializable
@SerialName("OneHot")
class OneHotEncoder(private val options: List<List<String>>): Transformer {
    private var totalDimension: Int = 0
    private var featureStarts: MutableList<Int> = mutableListOf()

    init {
        val dimensions = options.map { it.size - 1 }
        featureStarts = MutableList(dimensions.size) { 0 }
        totalDimension = dimensions.sum()
        for (i in 1..dimensions.lastIndex)
            featureStarts[i] = featureStarts[i - 1] + dimensions[i - 1]
    }

    override fun transform(features: List<FeatureValue>): List<FeatureValue> {
        val encoding = MutableList(totalDimension) { ONE_HOT_FALSE }
        for (i in 0..options.lastIndex) {
            val featureValueId = options[i].indexOf(features[i].toString())
            if (featureValueId > 0)
                encoding[featureStarts[i] + featureValueId - 1] = ONE_HOT_TRUE
        }
        return encoding
    }
}
