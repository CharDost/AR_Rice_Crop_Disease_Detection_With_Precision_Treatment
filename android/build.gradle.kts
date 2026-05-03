// Top-level build file for Rice Disease Detector
// Project: AR Rice Crop Disease Detection with Precision Treatment

plugins {
    id("com.android.application") version "8.2.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.21" apply false
    id("com.google.devtools.ksp") version "1.9.21-1.0.15" apply false
}

tasks.register("clean", Delete::class) {
    delete(layout.buildDirectory)
}
