package org.titanic_test

import kotlinx.serialization.modules.SerializersModule

val PIPELINE_SERIALIZATION_CONTEXT = SerializersModule {
    polymorphic(FeatureValue::class) {
        FloatValue::class with FloatValue.serializer()
        StringValue::class with StringValue.serializer()
        LongValue::class with LongValue.serializer()
    }
    polymorphic(Transformer::class) {
        TransformerPipeline::class with TransformerPipeline.serializer()
        OneHotEncoder::class with OneHotEncoder.serializer()
        Imputer::class with Imputer.serializer()
        Scaler::class with Scaler.serializer()
    }
    polymorphic(Predictor::class) {
        LogisticRegressionModel::class with LogisticRegressionModel.serializer()
    }
}