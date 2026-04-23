import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Dense(embed_dim)
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        # Extract patches
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # Reshape to (batch, num_patches, patch_size*patch_size*channels)
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        # Project to embedding dimension
        return self.projection(patches)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])
        
    def call(self, x, training):
        # Multi-head attention with residual
        attn_output = self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + attn_output
        # MLP with residual
        mlp_output = self.mlp(self.norm2(x), training=training)
        return x + mlp_output

def create_vit_model(
    input_shape=(16, 22, 1),
    patch_size=2,
    num_patches=None,
    embed_dim=64,
    num_heads=4,
    mlp_dim=128,
    num_blocks=4,
    dropout=0.1,
    num_classes=1
):
    inputs = layers.Input(shape=input_shape)
    
    # Patch embedding
    patches = PatchEmbedding(patch_size, embed_dim)(inputs)
    
    # Calculate number of patches
    if num_patches is None:
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    
    # Add positional embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(positions)
    patches = patches + pos_embedding
    
    # Add class token
    class_token = tf.Variable(tf.random.normal([1, 1, embed_dim]), trainable=True)
    class_tokens = tf.broadcast_to(class_token, [tf.shape(patches)[0], 1, embed_dim])
    patches = tf.concat([class_tokens, patches], axis=1)
    
    # Transformer blocks
    x = patches
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)(x, training=True)
    
    # Extract class token and classify
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0]  # Take the class token
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# Create and compile the ViT model
model_vit = create_vit_model(
    input_shape=(16, 22, 1),
    patch_size=2,  # 2x2 patches -> 8x11 = 88 patches
    embed_dim=64,
    num_heads=4,
    mlp_dim=128,
    num_blocks=4,
    dropout=0.25
)

model_vit.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model_vit.summary()

# Train the model
history_vit = model_vit.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=32,
    shuffle=True,
    verbose=1
)
