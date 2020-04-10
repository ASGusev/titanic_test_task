package org.titanic_test

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.math.exp


private const val PROBABILITY_THRESHOLD = .5F
private const val LABEL_FALSE = 0L
private const val LABEL_TRUE = 1L


private fun sigmoid(x: Float) = 1 / (1 + exp(-x))


@Serializable
@SerialName("LogisticRegression")
class LogisticRegressionModel(private val coefficients: List<Float>, private val intercept: Float): Predictor {
    override fun predictProbability(features: List<FeatureValue>): Float {
        var weightedFeatureSum = intercept
        for (i in 0..coefficients.lastIndex)
            weightedFeatureSum += coefficients[i] * features[i].floatValue
        return sigmoid(weightedFeatureSum)
    }

    override fun predictClass(features: List<FeatureValue>) =
        if (predictProbability(features) > PROBABILITY_THRESHOLD)
            LABEL_TRUE
        else
            LABEL_FALSE
}
