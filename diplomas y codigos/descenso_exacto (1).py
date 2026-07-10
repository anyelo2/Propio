import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from google.colab import drive

# 1. Montar Google Drive
drive.mount('/content/drive')

# 2. Cargar datos
def load_features():
    data_path = '/content/drive/MyDrive/raw_features.npz'
    data = np.load(data_path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

# 3. Funciones auxiliares
def sigmoid(z):
    return 1 / (1 + tf.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))


def train_logistic_regression_corrected(X_train, y_train, X_test, y_test, max_epochs=200000):
    X_train_tf = tf.constant(X_train, dtype=tf.float32)
    y_train_tf = tf.constant(y_train, dtype=tf.float32)
    X_test_tf = tf.constant(X_test, dtype=tf.float32)

    n_features = X_train.shape[1]
    beta = tf.Variable(tf.random.normal(shape=(n_features,), stddev=0.01), dtype=tf.float32)
    beta_0 = tf.Variable(0.0, dtype=tf.float32)

    grad_norms = []
    losses = []
    accuracies = []
    alphas = []
    param_diffs = []

    print("Iteración\t||∇f(xₖ)||\tPérdida\t\tExactitud\tαₖ\t\t||xₖ₊₁-xₖ||")
    print("------------------------------------------------------------------")

    for epoch in range(max_epochs):
        with tf.GradientTape() as tape:
            z = beta_0 + tf.tensordot(X_train_tf, beta, axes=1)
            p = sigmoid(z)
            current_loss = binary_cross_entropy(y_train_tf, p)

        grads = tape.gradient(current_loss, [beta_0, beta])
        g = tf.concat([tf.reshape(grads[0], [1]), grads[1]], axis=0)
        grad_norm = tf.norm(g).numpy()

        current_params = np.concatenate([[beta_0.numpy()], beta.numpy()])

        def loss_func(alpha):
            new_beta_0 = beta_0.numpy() + alpha * -grads[0].numpy()
            new_beta = beta.numpy() + alpha * -grads[1].numpy()
            z_new = new_beta_0 + X_train_tf.numpy().dot(new_beta)
            p_new = 1 / (1 + np.exp(-z_new))
            return -np.mean(y_train_tf.numpy() * np.log(p_new + 1e-7) + (1-y_train_tf.numpy()) * np.log(1-p_new + 1e-7))

        res = minimize_scalar(loss_func, bounds=(0, 2), method='bounded')
        alpha = res.x
        new_loss = res.fun

        beta_0.assign_add(alpha * -grads[0])
        beta.assign_add(alpha * -grads[1])

        new_params = np.concatenate([[beta_0.numpy()], beta.numpy()])
        param_diff = np.linalg.norm(new_params - current_params)

        test_pred = (sigmoid(beta_0 + tf.tensordot(X_test_tf, beta, axes=1)) > 0.5).numpy()
        test_acc = accuracy_score(y_test, test_pred)

        grad_norms.append(grad_norm)
        losses.append(current_loss.numpy())
        accuracies.append(test_acc)
        alphas.append(alpha)
        param_diffs.append(param_diff)

        if epoch % 5000 == 0:
            print(f"{epoch}\t{grad_norm:.4e}\t{current_loss.numpy():.6f}\t{test_acc:.4f}\t\t{alpha:.4e}\t{param_diff:.4e}")

        if epoch > 0 and losses[-1] >= losses[-2]:
            print("\n*** Criterio de parada activado: f(xₖ₊₁) ≥ f(xₖ) ***")
            print(f"Iteración {epoch}: Pérdida actual = {losses[-1]:.6f}, Pérdida anterior = {losses[-2]:.6f}")
            break

    print("\n=== Resultados del entrenamiento ===")
    print(f"Iteraciones realizadas: {len(losses)}")
    print(f"Última pérdida: {losses[-1]:.6f}")
    print(f"Última exactitud: {accuracies[-1]:.4f}")
    print(f"Última norma del gradiente: {grad_norms[-1]:.4e}")

    print("\n=== OPTIMAL RESULTS ===")
    optimal_loss = losses[-1]
    print(f"\nOptimal objective function value (loss): {optimal_loss:.6f}")
    print("\nOptimal decision variables:")
    print(f"Bias term (beta_0): {beta_0.numpy():.6f}")
    print("\nWeight coefficients (beta):")
    for i, val in enumerate(beta.numpy()[:10]):
        print(f"  beta[{i}] = {val:.6f}")
    if len(beta.numpy()) > 10:
        print(f"  ... and {len(beta.numpy()) - 10} more weights.")

    # Mostrar evolución de la norma del gradiente
    plt.figure(figsize=(10, 5))
    plt.plot(grad_norms)
    plt.title('Evolución de la norma del gradiente')
    plt.xlabel('Iteración')
    plt.ylabel('||∇f(xₖ)||')
    plt.grid(True)
    plt.show()

    # Matriz de confusión
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de confusión')
    plt.colorbar()
    plt.xticks([0, 1], ['No cáncer', 'Cáncer'])
    plt.yticks([0, 1], ['No cáncer', 'Cáncer'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')

    plt.tight_layout()
    plt.show()

    return beta_0.numpy(), beta.numpy(), losses

# 5. Ejecución principal
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_features()
    print("Datos cargados exitosamente")
    print(f"Clases en entrenamiento: {np.bincount(y_train.astype(int))}")
    print(f"Clases en prueba: {np.bincount(y_test.astype(int))}")

    print("\nComenzando entrenamiento...")
    beta_0, beta, losses = train_logistic_regression_corrected(X_train, y_train, X_test, y_test, max_epochs=200000)