package org.titanic_test

import kotlinx.serialization.Serializable


private const val ONE_HOT_TRUE = 1F
private const val ONE_HOT_FALSE = 0F


@Serializable
class FillerFloat(private val values: List<Float>) {
    fun transform(features: List<String>): List<Float> = features
        .zip(values)
        .map { (featureString, fillingValue) ->
            if (featureString.isEmpty())
                return@map fillingValue
            val featureValue = featureString.toFloat()
            return@map if (featureValue.isNaN()) fillingValue else featureValue
        }
}


@Serializable
class FillerString(private val values: List<String>) {
    fun transform(features: List<String>): List<String> = features
        .zip(values)
        .map { (f, v) -> if (f.isEmpty()) v else f }
}


@Serializable
class Scaler(private val min: List<Float>, private val range: List<Float>) {
    fun transform(features: List<Float>): List<Float> {
        val scaledFeatures = MutableList(features.size) { 0F }
        for (i in 0..features.lastIndex)
            scaledFeatures[i] = (features[i] - min[i]) / range[i]
        return scaledFeatures
    }
}


@Serializable
class OneHotEncoder(private val options: List<List<String>>) {
    private var totalDimension: Int = 0
    private var featureStarts: MutableList<Int> = mutableListOf()

    init {
        val dimensions = options.map { it.size - 1 }
        featureStarts = MutableList(dimensions.size) { 0 }
        totalDimension = dimensions.sum()
        for (i in 1..dimensions.lastIndex)
            featureStarts[i] = featureStarts[i - 1] + dimensions[i - 1]
    }

    fun transform(features: List<String>): List<Float> {
        val encoding = MutableList(totalDimension) { ONE_HOT_FALSE }
        for (i in 0..options.lastIndex) {
            val featureValueId = options[i].indexOf(features[i])
            if (featureValueId > 0)
                encoding[featureStarts[i] + featureValueId - 1] = ONE_HOT_TRUE
        }
        return encoding
    }
}
