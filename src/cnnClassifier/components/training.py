from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, f1_score,recall_score
from tensorflow.keras.callbacks import Callback

class MetricsCallback(Callback):
    def __init__(self, validation_generator):
        super().__init__()
        self.validation_generator = validation_generator

    def on_epoch_end(self, epoch, logs=None):
        val_steps = len(self.validation_generator)
        y_true = []
        y_pred = []
        
        # Get true labels and predicted labels for validation data
        for i in range(val_steps):
            x_val, y_val = next(self.validation_generator)
            y_pred_batch = np.argmax(self.model.predict(x_val), axis=1)
            y_true_batch = np.argmax(y_val, axis=1)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        precision = precision_score(y_true, y_pred,average="micro")
        f1 = f1_score(y_true, y_pred,average='micro')
        recall = recall_score(y_true, y_pred,average="micro")
        
        print(f'Precision: {precision:.4f} - F1 Score: {f1:.4f} - Recall Score: {recall:.4f}')


class Training :
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator


        self.train_generator=train_datagenerator.flow_from_directory(
            directory =self.config.training_data,
            subset ="training",
            shuffle=True,
            **dataflow_kwargs
        )
        
        # Access the class indices mapping
        class_indices = self.train_generator.class_indices

        # Invert the dictionary to get numerical labels to class names
        numerical_to_classes = {v: k for k, v in class_indices.items()}

        # Print the numerical label to class name mapping
        print("Numerical label to class name mapping:")
        for numerical_label, class_name in numerical_to_classes.items():
            print(f"Numerical Label: {numerical_label} -> Class Name: {class_name}")

    
    @staticmethod
    def save_model(path : Path , model : tf.keras.Model):
        model.save(path)
        print("Trained model is saved")


    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )