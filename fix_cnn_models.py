"""
Fix CNN Model Loading Issues
Run this once to re-save your models properly
"""

import tensorflow as tf
import numpy as np

print("🔄 Fixing CNN Regression Model...")
try:
    # Load the old model
    model = tf.keras.models.load_model("singing_skill_model.h5", compile=False)
    
    # Recompile properly
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Save in new format
    model.save("singing_skill_model_fixed.keras")
    print("✅ Fixed CNN Regression model saved as singing_skill_model_fixed.keras")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🔄 Fixing CNN Classification Model...")
try:
    # Load the old model
    model = tf.keras.models.load_model("singing_skill_classification_model.h5", compile=False)
    
    # Recompile properly
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save in new format
    model.save("singing_skill_classification_model_fixed.keras")
    print("✅ Fixed CNN Classification model saved as singing_skill_classification_model_fixed.keras")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🎯 Done! Now update app.py to use the fixed models.")