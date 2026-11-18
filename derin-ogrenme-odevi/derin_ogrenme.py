import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

# ==========================================
# 3. BOLUM: KONVOLUSYON VE FILTRE GORSELLESTIRME
# ==========================================
conv = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer='he_normal'
)

dummy_input = tf.random.normal((1, 128, 128, 1))
output_conv = conv(dummy_input)
weights, biases = conv.get_weights()

print("--- 3. BOLUM: KONVOLUSYON ---")
print("Giris boyutu:", dummy_input.shape)
print("Cikis boyutu:", output_conv.shape)
print("Filtre agirliklarinin boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)
print("Ilk filtre agirliklari:\n", weights[:, :, 0, 0])

plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(weights[:, :, 0, i], cmap='viridis')
    plt.title(f'Filtre {i+1}')
    plt.axis('off')
plt.suptitle("16 Filtrenin Baslangic Agirliklari")
plt.tight_layout()
plt.savefig("filtreler.png", dpi=150)
plt.close()

# ==========================================
# 4. BOLUM: MAX POOLING ISLEMI
# ==========================================
input_data = tf.constant([
    [[1., 3., 2., 4.],
     [5., 6., 1., 2.],
     [7., 2., 8., 3.],
     [4., 1., 5., 9.]]
])
input_data = tf.reshape(input_data, (1, 4, 4, 1))

pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
output_pool = pool(input_data)

print("\n--- 4. BOLUM: POOLING ---")
print("Giris:\n", input_data.numpy()[0, :, :, 0])
print("Pooling sonucu:\n", output_pool.numpy()[0, :, :, 0])

# ==========================================
# 5. BOLUM: RELU VE SOFTMAX
# ==========================================
x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
relu = tf.keras.layers.ReLU()
y = relu(x)

logits = tf.constant([[2.0, 1.0, 0.1]])
softmax = tf.keras.layers.Softmax()
probabilities = softmax(logits)

print("\n--- 5. BOLUM: RELU & SOFTMAX ---")
print("ReLU giris:", x.numpy())
print("ReLU cikis:", y.numpy())
print("Softmax sonuclari:", probabilities.numpy())
print("Olasiliklarin toplami:", tf.reduce_sum(probabilities).numpy())

# ==========================================
# 6. VE 8. BOLUM: FASHION-MNIST CNN VE DENEYLER
# ==========================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

def deney_calistir(filtre_sayisi):
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filtre_sayisi, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    baslangic = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    sure = time.time() - baslangic
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    train_acc = history.history['accuracy'][-1]
    return model, history, train_acc, test_loss, test_acc, sure

print("\n--- DENEY 1 (8 FILTRE) ---")
model1, hist1, tr_acc1, te_loss1, te_acc1, sure1 = deney_calistir(8)
model1.summary()

print("\n--- DENEY 2 (16 FILTRE) ---")
model2, hist2, tr_acc2, te_loss2, te_acc2, sure2 = deney_calistir(16)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist1.history['accuracy'], label='Training')
plt.plot(hist1.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist1.history['loss'], label='Training')
plt.plot(hist1.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("egitim_grafikleri.png", dpi=150)
plt.close()

print("\n================ KARSILASTIRMA TABLOSU ================")
print(f"Deney 1 (8 Filtre)  -> Egitim Acc: %{tr_acc1*100:.2f} | Test Acc: %{te_acc1*100:.2f} | Test Loss: {te_loss1:.4f} | Sure: {sure1:.2f} sn")
print(f"Deney 2 (16 Filtre) -> Egitim Acc: %{tr_acc2*100:.2f} | Test Acc: %{te_acc2*100:.2f} | Test Loss: {te_loss2:.4f} | Sure: {sure2:.2f} sn")