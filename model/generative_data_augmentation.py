"""
Generative Data Augmentation for EEG Data
Uses Variational Autoencoder (VAE) and GAN to generate synthetic EEG samples
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class VAEGenerator:
    """Variational Autoencoder for EEG data generation"""
    
    def __init__(self, input_dim, latent_dim=20, intermediate_dim=64):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        """Build VAE architecture"""
        # Encoder
        encoder_inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(self.intermediate_dim, activation='relu')(encoder_inputs)
        x = layers.Dense(self.intermediate_dim // 2, activation='relu')(x)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        class SamplingLayer(layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = SamplingLayer()([z_mean, z_log_var])
        
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.intermediate_dim // 2, activation='relu')(latent_inputs)
        x = layers.Dense(self.intermediate_dim, activation='relu')(x)
        decoder_outputs = layers.Dense(self.input_dim, activation='linear')(x)
        
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
        
        # VAE with custom training step
        class VAE(keras.Model):
            def __init__(self, encoder, decoder, **kwargs):
                super(VAE, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
                self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
                self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
                self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            
            @property
            def metrics(self):
                return [
                    self.total_loss_tracker,
                    self.reconstruction_loss_tracker,
                    self.kl_loss_tracker,
                ]
            
            def train_step(self, data):
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = self.encoder(data)
                    reconstruction = self.decoder(z)
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            keras.losses.mse(data, reconstruction), axis=1
                        )
                    )
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                    total_loss = reconstruction_loss + kl_loss
                
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                self.total_loss_tracker.update_state(total_loss)
                self.reconstruction_loss_tracker.update_state(reconstruction_loss)
                self.kl_loss_tracker.update_state(kl_loss)
                return {
                    "loss": self.total_loss_tracker.result(),
                    "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                    "kl_loss": self.kl_loss_tracker.result(),
                }
        
        self.vae = VAE(self.encoder, self.decoder)
        self.vae.compile(optimizer=keras.optimizers.Adam())
        
        return self.vae
    
    def train(self, X, epochs=100, batch_size=32, verbose=1):
        """Train the VAE"""
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model if not already built
        if self.vae is None:
            self.build_model()
        
        # Adjust batch size if dataset is too small
        batch_size = min(batch_size, len(X_scaled))
        if batch_size < 2:
            batch_size = 2
        
        # Train
        history = self.vae.fit(
            X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True
        )
        
        return history
    
    def generate(self, n_samples=100):
        """Generate synthetic samples"""
        # Sample from latent space
        z_samples = np.random.normal(0, 1, (n_samples, self.latent_dim))
        
        # Decode
        generated_scaled = self.decoder.predict(z_samples, verbose=0)
        
        # Inverse transform
        generated = self.scaler.inverse_transform(generated_scaled)
        
        return generated


class GANGenerator:
    """Generative Adversarial Network for EEG data generation"""
    
    def __init__(self, input_dim, latent_dim=50):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.scaler = StandardScaler()
        
    def build_generator(self):
        """Build generator network"""
        model = keras.Sequential([
            layers.Dense(128, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(self.input_dim, activation='linear')
        ])
        return model
    
    def build_discriminator(self):
        """Build discriminator network"""
        model = keras.Sequential([
            layers.Dense(512, input_dim=self.input_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(128),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_gan(self):
        """Build GAN model"""
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Freeze discriminator for generator training
        self.discriminator.trainable = False
        
        # GAN model
        z = layers.Input(shape=(self.latent_dim,))
        generated = self.generator(z)
        validity = self.discriminator(generated)
        
        self.gan = keras.Model(z, validity)
        self.gan.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        return self.gan
    
    def train(self, X, epochs=100, batch_size=32, verbose=1):
        """Train the GAN"""
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build models if not already built
        if self.gan is None:
            self.build_gan()
        
        # Labels
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X_scaled.shape[0], batch_size)
            real_samples = X_scaled[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_samples = self.generator.predict(noise, verbose=0)
            
            d_loss_real = self.discriminator.train_on_batch(real_samples, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, valid)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
    
    def generate(self, n_samples=100):
        """Generate synthetic samples"""
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated_scaled = self.generator.predict(noise, verbose=0)
        generated = self.scaler.inverse_transform(generated_scaled)
        return generated


class EEGDataAugmenter:
    """Main class for EEG data augmentation"""
    
    def __init__(self, method='vae'):
        """
        Initialize augmenter
        Args:
            method: 'vae' or 'gan'
        """
        self.method = method
        self.generator = None
        
    def fit(self, X, epochs=100, batch_size=32, verbose=1):
        """Fit the generative model"""
        input_dim = X.shape[1]
        
        # Adjust batch size for small datasets
        effective_batch_size = min(batch_size, max(2, len(X) // 2))
        
        try:
            if self.method == 'vae':
                self.generator = VAEGenerator(input_dim=input_dim)
                self.generator.train(X, epochs=epochs, batch_size=effective_batch_size, verbose=verbose)
            elif self.method == 'gan':
                self.generator = GANGenerator(input_dim=input_dim)
                self.generator.train(X, epochs=epochs, batch_size=effective_batch_size, verbose=verbose)
            else:
                raise ValueError("Method must be 'vae' or 'gan'")
        except Exception as e:
            if verbose:
                print(f"Warning: Generative model training failed: {e}")
                print("Falling back to statistical augmentation...")
            # Fallback to statistical augmentation
            self.generator = None
            self._statistical_augmentation = True
            self._X_train = X
        
        return self
    
    def generate(self, n_samples=100):
        """Generate synthetic samples"""
        if self.generator is None:
            # Use statistical augmentation as fallback
            if hasattr(self, '_statistical_augmentation') and self._statistical_augmentation:
                return self._statistical_generate(n_samples)
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.generator.generate(n_samples=n_samples)
    
    def _statistical_generate(self, n_samples):
        """Generate samples using statistical methods (fallback)"""
        # Use Gaussian noise with same mean and covariance as training data
        mean = np.mean(self._X_train, axis=0)
        cov = np.cov(self._X_train.T)
        
        # Add small regularization to covariance matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        
        try:
            generated = np.random.multivariate_normal(mean, cov, n_samples)
        except:
            # If covariance matrix is not positive definite, use simpler method
            std = np.std(self._X_train, axis=0)
            generated = np.random.normal(mean, std, (n_samples, len(mean)))
        
        return generated
    
    def augment_class(self, X_class, n_samples, epochs=100, batch_size=32, verbose=1):
        """Augment data for a specific class"""
        print(f"\nAugmenting class with {len(X_class)} samples to {n_samples} samples...")
        
        # Convert to numpy if DataFrame
        if isinstance(X_class, pd.DataFrame):
            X_class_array = X_class.values
            columns = X_class.columns
            is_dataframe = True
        else:
            X_class_array = X_class
            is_dataframe = False
        
        # Reset generator state
        self.generator = None
        if hasattr(self, '_statistical_augmentation'):
            delattr(self, '_statistical_augmentation')
        
        # Fit generator on class data
        self.fit(X_class_array, epochs=epochs, batch_size=min(batch_size, len(X_class_array)), verbose=verbose)
        
        # Generate synthetic samples
        n_to_generate = max(0, n_samples - len(X_class_array))
        if n_to_generate > 0:
            try:
                synthetic = self.generate(n_samples=n_to_generate)
            except Exception as e:
                if verbose:
                    print(f"Warning: Generation failed, using statistical method: {e}")
                # Force statistical augmentation
                self._statistical_augmentation = True
                self._X_train = X_class_array
                synthetic = self.generate(n_samples=n_to_generate)
            
            # Combine real and synthetic
            augmented = np.vstack([X_class_array, synthetic])
            
            if is_dataframe:
                augmented = pd.DataFrame(augmented[:n_samples], columns=columns)
            else:
                augmented = augmented[:n_samples]
            
            return augmented
        else:
            return X_class


def augment_dataset(X_dict, target_samples_per_class=100, method='vae', epochs=100, verbose=1):
    """
    Augment entire dataset
    
    Args:
        X_dict: Dictionary with class labels as keys and data arrays as values
        target_samples_per_class: Target number of samples per class
        method: 'vae' or 'gan'
        epochs: Training epochs
        verbose: Verbosity level
    
    Returns:
        Augmented dataset dictionary
    """
    augmenter = EEGDataAugmenter(method=method)
    augmented_dict = {}
    
    for class_label, X_class in X_dict.items():
        print(f"\n{'='*60}")
        print(f"Augmenting class: {class_label}")
        print(f"{'='*60}")
        
        if len(X_class) >= target_samples_per_class:
            print(f"Class already has {len(X_class)} samples (target: {target_samples_per_class})")
            augmented_dict[class_label] = X_class
        else:
            augmented = augmenter.augment_class(
                X_class, 
                target_samples_per_class, 
                epochs=epochs, 
                verbose=verbose
            )
            augmented_dict[class_label] = augmented
            print(f"Generated {len(augmented) - len(X_class)} synthetic samples")
    
    return augmented_dict


if __name__ == "__main__":
    # Example usage
    print("EEG Data Augmentation Module")
    print("This module provides VAE and GAN-based data augmentation for EEG data")

