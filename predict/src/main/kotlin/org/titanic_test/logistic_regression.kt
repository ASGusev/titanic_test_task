package org.titanic_test

import kotlinx.serialization.Serializable
import kotlin.math.exp


private const val PROBABILITY_THRESHOLD = .5F
private const val LABEL_FALSE: Int = 0
private const val LABEL_TRUE: Int = 1


private fun sigmoid(x: Float) = 1 / (1 + exp(-x))


@Serializable
class LogisticRegressionModel(private val coefficients: List<Float>, private val intercept: Float) {
    fun predictProbability(features: List<Float>): Float {
        var weightedFeatureSum = intercept
        for (i in 0..coefficients.lastIndex)
            weightedFeatureSum += coefficients[i] * features[i]
        return sigmoid(weightedFeatureSum)
    }

    fun predict(features: List<Float>) =
        if (predictProbability(features) > PROBABILITY_THRESHOLD)
            LABEL_TRUE
        else
            LABEL_FALSE
}
