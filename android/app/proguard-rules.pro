# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Keep TensorFlow Lite classes
-keep class org.tensorflow.** { *; }
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.**

# Keep data classes
-keep class com.ricedisease.detector.data.** { *; }
-keep class com.ricedisease.detector.inference.** { *; }

# Keep Kotlin data classes
-keepclassmembers class * {
    @com.google.gson.annotations.SerializedName <fields>;
}
