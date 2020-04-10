package org.titanic_test

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable


@Serializable
sealed class FeatureValue {
    abstract val isEmpty: Boolean
    abstract val floatValue: Float
}

@Serializable
@SerialName("FloatValue")
data class FloatValue(val value: Float): FeatureValue() {
    override val isEmpty = value.isNaN()
    override val floatValue
        get() = value
    override fun toString() = value.toString()
}

@Serializable
@SerialName("StringValue")
data class StringValue(val value: String): FeatureValue() {
    override val isEmpty = value.isEmpty()
    override val floatValue
        get() = value.toFloat()
    override fun toString() = value
}

@Serializable
@SerialName("LongValue")
data class LongValue(val value: Long): FeatureValue() {
    override val isEmpty = false
    override val floatValue
        get() = value.toFloat()
    override fun toString() = value.toString()
}