import numpy as np
import tensorflow as tf
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


def train_logistic_regression_backtracking(X_train, y_train, X_test, y_test,
                                         c1=0.3, rho=0.5, max_epochs=1000):
    X_train_tf = tf.constant(X_train, dtype=tf.float32)
    y_train_tf = tf.constant(y_train, dtype=tf.float32)
    X_test_tf = tf.constant(X_test, dtype=tf.float32)

    # Initialize parameters - Adding beta_0 to the beta variable for simpler dot product
    n_features = X_train.shape[1]
    # Initialize beta with bias term
    beta = tf.Variable(tf.random.normal(shape=(n_features + 1,), stddev=0.01), dtype=tf.float32)


    # Tracking variables
    grad_norms = []
    train_losses = []
    test_accuracies = []
    step_sizes = []

    print("k\t||∇f(xₖ)||\tPérdida\t\tExactitud\tαₖ")
    print("--------------------------------------------------")

    # Add bias term to X_train for simplified calculation
    X_train_aug = tf.concat([tf.ones((X_train.shape[0], 1), dtype=tf.float32), X_train_tf], axis=1)
    X_test_aug = tf.concat([tf.ones((X_test.shape[0], 1), dtype=tf.float32), X_test_tf], axis=1)


    for epoch in range(max_epochs):
        with tf.GradientTape() as tape:
            # Forward pass using augmented X
            z = tf.tensordot(X_train_aug, beta, axes=1)
            p = sigmoid(z)
            current_loss = binary_cross_entropy(y_train_tf, p)

        # Calcular gradiente
        grads = tape.gradient(current_loss, beta) # Gradient with respect to augmented beta
        grad_norm = tf.norm(grads).numpy()
        grad_norms.append(grad_norm)

        # Dirección de descenso
        direction = -grads  # Negative gradient

        
        alpha = 1.0
        # print(f"Epoch {epoch}
        for backtrack_step in range(20):  # Máximo 20 intentos de backtracking
            beta_proposed = beta + alpha * direction

            with tf.GradientTape() as tape_new:
                z_new = tf.tensordot(X_train_aug, beta_proposed, axes=1)
                p_new = sigmoid(z_new)
                new_loss = binary_cross_entropy(y_train_tf, p_new)

            # print(f"  Backtrack step {backtrack_step}: alpha={alpha}, new_loss={new_loss.numpy():.6f}") # Debug print
            # Armijo condition
            directional_derivative = tf.tensordot(grads, direction, axes=1) # Dot product of gradient and direction
            armijo_condition = current_loss + c1 * alpha * directional_derivative

            # print(f"  Backtrack step {backtrack_step}: Armijo condition value={armijo_condition.numpy():.6f}") # Debug print

            if new_loss <= armijo_condition:
                # print("  Armijo condition met.") # Debug print
                break
            alpha *= rho
            # print(f"  Reducing alpha to {alpha}") # Debug print

        # Actualizar parámetros
        beta.assign_add(alpha * direction)
        step_sizes.append(alpha)

        # Calcular exactitud en test
        test_pred = (sigmoid(tf.tensordot(X_test_aug, beta, axes=1)) > 0.5).numpy()
        test_acc = accuracy_score(y_test, test_pred)

        # Guardar métricas
        train_losses.append(current_loss.numpy())
        test_accuracies.append(test_acc)

        # Imprimir progreso cada 100 iteraciones
        if epoch % 5000 == 0:
            print(f"{epoch}\t{grad_norm:.4e}\t{current_loss.numpy():.6f}\t{test_acc:.4f}\t\t{alpha:.4e}")

        # Criterio de parada principal
        if epoch > 0 and train_losses[-1] >= train_losses[-2]:
            print("\n*** Criterio de parada: f(xₖ₊₁) ≥ f(xₖ) ***")
            print(f"Iteración {epoch}: Pérdida actual = {train_losses[-1]:.6f}, Pérdida anterior = {train_losses[-2]:.6f}")
            break

    # Resultados finales
    print("\n=== Resultados del entrenamiento ===")
    print(f"Iteraciones realizadas: {len(train_losses)}")
    print(f"Última pérdida: {train_losses[-1]:.6f}")
    print(f"Última exactitud: {test_accuracies[-1]:.4f}")
    print(f"Última norma del gradiente: {grad_norms[-1]:.4e}")


    print("\n=== OPTIMAL RESULTS ===")

    # 1. Optimal objective function value
    optimal_loss = train_losses[-1]
    print(f"\nOptimal objective function value (loss): {optimal_loss:.6f}")

    # 2. Optimal decision variables
    print("\nOptimal decision variables:")
    # Extract bias term and weights from the augmented beta
    optimal_beta_0 = beta.numpy()[0]
    optimal_weights = beta.numpy()[1:]

    print(f"Bias term (beta_0): {optimal_beta_0:.6f}")
    print("\nWeight coefficients (beta):")
    # Print only a few for brevity if many features
    for i, val in enumerate(optimal_weights[:10]): # Print first 10 weights as example
        print(f"  beta[{i}] = {val:.6f}")
    if len(optimal_weights) > 10:
        print(f"  ... and {len(optimal_weights) - 10} more weights.")


    # 1. Gráfica de la norma del gradiente vs iteraciones
    plt.figure(figsize=(10, 5))
    plt.plot(grad_norms, 'b-', linewidth=2)
    plt.title('Evolución de la Norma del Gradiente', fontsize=14)
    plt.xlabel('Iteración (k)', fontsize=12)
    plt.ylabel('||∇f(xₖ)||', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # 2. Matriz de confusión
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['no_cancer', 'cancer'], rotation=45)
    plt.yticks(tick_marks, ['no_cancer', 'cancer'])
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Etiqueta verdadera', fontsize=12)

    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

    # Return beta including the bias term
    return beta.numpy()

# 5. Ejecución principal
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_features()
    print("\nDatos cargados exitosamente")
    print(f"Distribución clases entrenamiento: {np.bincount(y_train.astype(int))}")
    print(f"Distribución clases prueba: {np.bincount(y_test.astype(int))}")

    print("\nIniciando entrenamiento con backtracking...")
    beta = train_logistic_regression_backtracking(
        X_train, y_train, X_test, y_test,
        c1=0.3, rho=0.5, max_epochs=200000
    )