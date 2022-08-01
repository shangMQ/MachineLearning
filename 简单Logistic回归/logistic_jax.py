"""
基于jax实现逻辑回归
"""
import jax
import jax.numpy as jnp
from jax import value_and_grad
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# 定义模型
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)


# Training loss is the negative log-likelihood of the training examples.
def loss(W, b, inputs, targets):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.mean(jnp.log(label_probs))

def train_step(W, b, X, y, learning_rate):
    loss_value, Wb_grad = value_and_grad(loss, (0, 1))(W, b, X, y)
    W -= learning_rate * Wb_grad[0]
    b -= learning_rate * Wb_grad[1]
    return loss_value, W, b

def fit(W, b, X, y, epochs=1, learning_rate=1e-2):
    losses = jnp.array([])
    for _ in range(epochs):
        l, W, b = train_step(W, b, X, y, learning_rate=learning_rate)
        losses = jnp.append(losses, l)
    return losses, W, b

def plot_losses(losses):
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("Loss Plot")
    plt.show()

def validate_model(W, b, X_test, y_test):
    y_pred = predict(W, b, X_test)
    return roc_auc_score(y_test, y_pred)

if __name__ == "__main__":
    # 加载数据
    X, y = load_breast_cancer(return_X_y=True)

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    # 初始化超参数
    w = jnp.zeros((X.shape[1],))
    b = 0.0

    # 加载使用jax实现的逻辑回归
    loss, final_w, final_b = fit(w, b, X_train, y_train, epochs=20, learning_rate=1e-2)

    print(final_b)
    # 绘制损失
    plot_losses(loss)
    print(f"loss = {loss}")

    # 查看模型在测试集上的效果
    score = validate_model(final_w, final_b, X_test, y_test)
    print(f"socre in validation = {score}")
